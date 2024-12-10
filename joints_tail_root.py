import numpy as np
from utils import vector_to_translation_matrix
import open3d as o3d

from utils import align_vectors_to_axes, find_closest_orthogonal, find_points_within_radius, generate_rotation_matrix
from skopt import gp_minimize
from scipy.optimize import basinhopping
from scipy.sparse.linalg import eigsh
from surface_detection_utils import PCDSurface, o3DSurface
from sklearn.decomposition import PCA
from skopt.space import Integer


class BallJoint:
    def __init__(self, position, target_position, x_vector, y_vector, z_vector):
        # x axis is rotation around length of spine, z axis is towards process tip, y is towards transverse process

        self.initial_position = np.array(position)
        self.initial_target_position = np.array(target_position)
        self.position = np.array(position)
        self.target_position = target_position
        self.initial_x_vector = np.copy(x_vector)
        self.initial_y_vector = np.copy(y_vector)
        self.initial_z_vector = np.copy(z_vector)
        self.x_vector = x_vector
        self.y_vector = y_vector
        self.z_vector = z_vector
        self.axis_rotation = np.array([0.0, 0.0, 0.0])

    def update_position(self, transform):
        points = self.position.reshape(1, -1)
        homogeneous = np.hstack([points, np.ones((len(points), 1))]).T
        transformed = (transform @ homogeneous).T[:, :3][0]
        vector_points = np.array(
            [self.x_vector + self.position, self.y_vector + self.position, self.z_vector + self.position])

        homogeneous_vecs = np.hstack(
            [vector_points, np.ones((len(vector_points), 1))]).T
        transformed_vecs = (transform @ homogeneous_vecs).T[:, :3]

        self.position[:] = transformed[:]
        self.x_vector, self.y_vector, self.z_vector = transformed_vecs - self.position
        target_points = self.target_position.reshape(1, -1)

        homogeneous_target = np.hstack(
            [target_points, np.ones((len(target_points), 1))]).T
        transformed_target = (transform @ homogeneous_target).T[:, :3][0]
        self.target_position[:] = transformed_target[:]

    def reset_joint_position(self):
        self.position[:] = self.initial_position[:]
        self.target_position[:] = self.initial_target_position[:]
        self.axis_rotation = np.array([0.0, 0.0, 0.0])
        self.x_vector[:] = self.initial_x_vector[:]
        self.y_vector[:] = self.initial_y_vector[:]
        self.z_vector[:] = self.initial_z_vector[:]

    def set_axis_rotation(self, rotation):
        self.axis_rotation[:] = rotation[:]

    def set_rotation(self, rotation):
        self.axis_rotation = rotation

    def constrain_rotation_matrix(self, axis_rotation):

        _, R_constrained = generate_rotation_matrix(
            axis_rotation)

        T = np.eye(4)
        T[0:3, 0:3] = R_constrained
        return T

    def to_rotation_only_matrix(self, rotation_constraint=None):

        constrained_matrix = self.constrain_rotation_matrix(
            self.axis_rotation)

        return constrained_matrix

    def get_closest_spline_point_vector(self):
        closest = np.argmin(
            np.sqrt(np.sum((self.spline_points - self.position)**2, axis=1)))
        vec = self.spline_points[closest] - self.position
        return vec / np.linalg.norm(vec)

    def get_axes(self):

        return self.x_vector, self.y_vector, self.z_vector


class Bones:
    def get_sandwich_transform(self, joint: BallJoint, rotation_constraint=None):
        # first bring the joint to origin, then align joint axes with world
        head_joint_translation = np.array(joint.position) * -1.0
        t1 = vector_to_translation_matrix(head_joint_translation)
        x1, y1, z1 = joint.get_axes()
        rotation_alignment = align_vectors_to_axes(x1, y1, z1)
        rotation = joint.to_rotation_only_matrix()
        inv_t1 = np.linalg.inv(t1)
        inv_rotation = np.linalg.inv(rotation_alignment)
        return inv_t1 @ inv_rotation @ rotation @ rotation_alignment @ t1


class Vertebrae(Bones):

    def get_joint_axes(self, joint: BallJoint):
        return joint.get_axes(self.spline_points)

    def initialize_joints(self, x_vector, y_vector, z_vector):

        if self.head_child is not None:
            self.head_joint = BallJoint(
                self.bone_positions[1], self.head_child.bone_positions[1], x_vector, y_vector, z_vector)
        else:
            self.head_joint = None

    def __init__(self, bone_positions, partition_map, upvector, forward_vector, vertices, head_child=None):
        self.bone_positions = bone_positions
        self.cluster_id = bone_positions[2]
        self.vertex_ids = partition_map[self.cluster_id]
        self.head_child = head_child
        self.parent = None
        self.upvector = upvector
        self.forward_vector = forward_vector
        self.x_vector, self.y_vector, self.z_vector = self.get_vertebra_axis(
            vertices)
        self.initialize_joints(self.x_vector, self.y_vector, self.z_vector)

    def closest_aligning(self, target_vector, principal_vectors):
        dot_product = np.abs(np.dot(principal_vectors, target_vector))
        best_fit = principal_vectors[np.argmax(dot_product)]
        if np.sign(np.dot(best_fit, target_vector)) == -1:
            best_fit = best_fit * -1
        return best_fit

    def get_vertebra_axis(self, vertices):
        rel_verts = vertices[self.vertex_ids]
        pca = PCA()
        pca.fit(rel_verts)
        principal_vectors = pca.components_
        x_vector = self.closest_aligning(
            self.forward_vector, principal_vectors)
        z_vector = self.closest_aligning(self.upvector, principal_vectors)
        y = np.cross(z_vector, x_vector)

        # Normalize y
        y = y / np.linalg.norm(y)

        return x_vector, y, z_vector

    def get_vertebrae_z_axes_rotation(self, points):
        # warning this only works if aligned on depth image
        # only need rotation component not transform as taking quantile
        pca = PCA()
        pca.fit(points)
        principal_vectors = pca.components_
        cross_product = np.abs(np.dot(principal_vectors, np.array([0, 0, 1])))
        z_vector = principal_vectors[np.argmax(cross_product)]
        global_z = np.array([0, 0, 1])
        rotation_axis = np.cross(z_vector, global_z)
        if np.allclose(rotation_axis, 0):
            if np.allclose(z_vector, global_z):
                return np.eye(3)  # No rotation needed
            else:
                # Vectors are anti-parallel, rotate 180° around any perpendicular axis
                return -np.eye(3)  # Or rotate around x or y axis

        # Normalize rotation axis
        rotation_axis = rotation_axis / np.linalg.norm(rotation_axis)

        # Get rotation angle
        cos_angle = np.dot(z_vector, global_z)
        angle = np.arccos(np.clip(cos_angle, -1.0, 1.0))

        # Rodrigues rotation formula
        K = np.array([
            [0, -rotation_axis[2], rotation_axis[1]],
            [rotation_axis[2], 0, -rotation_axis[0]],
            [-rotation_axis[1], rotation_axis[0], 0]
        ])

        R = np.eye(3) + np.sin(angle) * K + (1 - cos_angle) * (K @ K)
        return R

    def get_tip_indices(self, vertices, tip_threshold):
        head_tip_ids = []
        vertex_points = vertices[self.vertex_ids]
        rotation = self.get_vertebrae_z_axes_rotation(vertex_points)

        transformed_subset = vertex_points @ rotation.T

        quantile = np.quantile(transformed_subset[:, 2], tip_threshold)
        # to map back from the subset we need to
        vertex_indices = list(np.array(self.vertex_ids)[np.argwhere(
            transformed_subset[:, 2] < quantile).reshape(-1)])

        if self.head_child is not None:
            head_tip_ids = self.head_child.get_tip_indices(
                vertices, tip_threshold)

        return vertex_indices + head_tip_ids

    def get_joints_and_child_joints(self):
        joints = []
        if self.head_joint:
            joints.append(self.head_joint)

        head_joints = []
        if self.head_child is not None:
            head_joints = self.head_child.get_joints_and_child_joints()

        return joints + head_joints

    def set_parents(self, parent=None):
        # this traverses and creates links back to parent
        self.parent = parent
        if self.head_child is None:
            return
        else:
            if self.head_child is not None:
                self.head_child.set_parents(parent=self)
            return

    def reset_joints(self):
        if self.head_joint is not None:
            self.head_joint.reset_joint_position()
        if self.head_child is not None:
            self.head_child.reset_joints()
        return

    # update joint params, propogate children position, then
    # to propogate transforms from current, I need to know, joint params,
    # qw, qx, qy, qz for each joint, as well as vertices affected by each joint, so for everything downstream, including joints bones vertices, etc
    # do translation from joint in question to coordinates origin, apply rotation, then back to
    def get_downstream_indices(self):

        head_ids = []
        if self.head_child is not None:
            head_ids = self.head_child.get_downstream_indices()

        return self.vertex_ids + head_ids

    @staticmethod
    def update_single_point_position(joint, transform):
        points = joint.position.reshape(1, -1)
        homogeneous = np.hstack([points, np.ones((len(points), 1))]).T
        transformed = (transform @ homogeneous).T[:, :3][0]
        joint.position = transformed

    def update_joint_positions(self, transform):
        # this should update the position of my joints, and any children I have

        if self.head_joint is not None:
            self.head_joint.update_position(transform)

        if self.head_child is not None:
            self.head_child.update_joint_positions(transform)

        return

    def propagate_joint_updates(self, spine):
        # each subject joint will have it's own affected vertices
        # first we need to update vertex and joint positions of child vertebrae affected by each joints
        # parameters, after which we then need to traverse down/up the column and
        if self.head_joint is not None:

            combined_head = self.get_sandwich_transform(
                self.head_joint)
            head_indices = self.head_child.get_downstream_indices()
            spine.update_vertex_positions(combined_head, head_indices)
            self.head_child.update_joint_positions(combined_head)
            self.head_joint.axis_rotation = np.array([0, 0, 0])
            self.head_child.propagate_joint_updates(spine)


class Spine(Bones):

    def __init__(self, root_vertebrae, vertices_array, video_curve_points, scene_points,
                 sampled_cloud, upvector,
                 translation_bound=5, alpha=0.5, beta=0.5, threshold=0.4, normal_cloud=None, tip_threshold=0.1):
        self.alpha = alpha
        self.beta = beta
        self.threshold = threshold
        self.root_vertebrae = root_vertebrae
        self.translation_offset = np.eye(4)
        self.initial_vertices = vertices_array.copy()
        self.vertices = vertices_array.copy()
        self.upvector = upvector

        self.initial_fitness = None
        self.video_curve_points = video_curve_points
        self.query_surface = PCDSurface(scene_points)
        self.evaluation = o3d.pipelines.registration.TransformationEstimationPointToPlane()

        self.all_joints = self.get_all_joints()

        self.rotation_bounds = [Integer(-50, 50)
                                for _ in range(0, (len(self.all_joints)) * 3)]

        for i in range(0, 3):

            self.rotation_bounds.append(Integer(-40, 40))

        # self.translation_bounds = [
        #     (-translation_bound, translation_bound) for _ in range(0, 3)]

        self.translation_bounds = [
            Integer(-3, 3) for _ in range(0, 3)]

        self.translation_bounds[-1] = Integer(-1, 1)
        # self.translation_bounds[-1] = (2, 3)
        # self.translation_bounds[0] = (-1.3, -1.2)

        self.id_transform = np.eye(4)
        self.get_centroid_joint_position_and_axis()

        self.process_indices = self.get_vertebrae_tip_indices(
            tip_threshold=tip_threshold)
        self.sampled_cloud = sampled_cloud
        temp_cloud = o3d.geometry.PointCloud()
        temp_cloud.points = o3d.utility.Vector3dVector(
            vertices_array[self.process_indices])
        self.process_cloud, _ = find_points_within_radius(
            self.sampled_cloud, temp_cloud, radius=4)

        self.process_pcd = o3d.geometry.PointCloud()
        self.process_pcd.points = o3d.utility.Vector3dVector(
            self.vertices[self.process_indices])
        self.normal_cloud = normal_cloud
        self.normal_query_surface = o3DSurface(self.process_cloud)

    def update_vertex_positions(self, transform, indices):
        if indices is not None:
            spine_coords = self.vertices[indices]
        else:
            spine_coords = self.vertices

        homogeneous = np.hstack([spine_coords, np.ones(
            (len(spine_coords), 1))]).T
        transformed = (transform @ homogeneous).T[:, :3]

        self.vertices[indices] = transformed

    def get_centroid_joint_position_and_axis(self):
        centroid_position = np.mean(self.initial_vertices, axis=0)

        pca = PCA()
        pca.fit(self.initial_vertices)
        principal_vectors = pca.components_

        root_vertebra_centroid = np.mean(
            self.initial_vertices[self.root_vertebrae.vertex_ids], axis=0)

        to_centroid_vector = centroid_position - root_vertebra_centroid
        to_centroid_vector = to_centroid_vector / \
            np.linalg.norm(to_centroid_vector)

        x_vector = self.root_vertebrae.closest_aligning(
            to_centroid_vector, principal_vectors)

        self.global_pca_x_vector = x_vector

        z_vector = self.root_vertebrae.closest_aligning(
            self.upvector, principal_vectors)
        y_vector = np.cross(x_vector, z_vector)
        self.centroid = BallJoint(
            centroid_position, centroid_position + x_vector, x_vector, y_vector, z_vector)

        return

    def get_vertebrae_tip_indices(self, tip_threshold):
        return self.root_vertebrae.get_tip_indices(self.vertices, tip_threshold)

    def get_tips(self, tip_threshold=0.1):
        tip_threshold = 1-tip_threshold
        head_joint_translation = np.array(self.centroid.position) * -1.0
        t1 = vector_to_translation_matrix(head_joint_translation)
        x1, y1, z1 = self.centroid.get_axes()
        rotation_alignment = align_vectors_to_axes(x1, y1, z1)
        combined = rotation_alignment @ t1

        homogeneous = np.hstack([self.vertices, np.ones(
            (len(self.vertices), 1))]).T
        transformed = (combined @ homogeneous).T[:, :3]
        quantile = np.quantile(transformed[:, 2], tip_threshold)
        return np.argwhere(transformed[:, 2] > quantile).reshape(-1)

    def get_tips_cloud(self, tip_threshold=0.1):
        quantile = np.quantile(np.asarray(self.sampled_cloud.points)[
                               :, 2], tip_threshold)
        return np.argwhere(np.asarray(self.sampled_cloud.points)[:, 2] < quantile).reshape(-1)

    def get_process_error(self, threshold=0.4):
        self.process_pcd.points = o3d.utility.Vector3dVector(
            self.vertices[self.process_indices])

        result = o3d.pipelines.registration.evaluate_registration(
            self.process_pcd, self.process_cloud, threshold, self.id_transform)
        if self.initial_fitness is None:
            self.initial_fitness = result.fitness

        return -len(np.asarray(result.correspondence_set))
        diff = self.initial_fitness - result.fitness
        return (result.inlier_rmse/threshold) + np.exp(diff)

        raw_error = 1 - result.fitness
        return raw_error
        return 1 - np.exp(-10 * raw_error)

    def get_surface_error(self):
        if self.normal_cloud is None:
            bounds = self.query_surface.check_point_position(self.vertices)
            raw_error = len(bounds[bounds == -1])/len(self.vertices)
        else:
            bounds = self.normal_query_surface.check_point_position(
                self.vertices[self.process_indices])
            raw_error = len(bounds[bounds == 1])  # /len(self.vertices)

        return raw_error
        # return 1 - np.exp(-10 * raw_error)

    def reset_spine(self):
        self.vertices[:, :] = self.initial_vertices[:, :]
        self.root_vertebrae.reset_joints()

    def get_all_joints(self):
        return self.root_vertebrae.get_joints_and_child_joints()

    def apply_joint_parameters(self, joint_parameters):
        # joint parameters is np.array of shape NumJointsx4
        for ind, joint in enumerate(self.all_joints):
            joint.set_axis_rotation(joint_parameters[ind*3: (ind*3)+3])

    def apply_joint_angles(self):
        self.root_vertebrae.propagate_joint_updates(self)

    def apply_global_transform(self):
        combined_transform = self.get_sandwich_transform(
            self.centroid, rotation_constraint=None)
        # rotation_constraint was (3.0, 3.0, 3.0)
        combined_rotation_translation = self.translation_offset @ combined_transform
        self.update_vertex_positions(combined_rotation_translation, None)
        all_joints = self.get_all_joints()
        for joint in all_joints:
            joint.update_position(combined_rotation_translation)

    def apply_global_parameters(self, global_parameters):

        self.centroid.set_axis_rotation(
            global_parameters[:3]
        )
        transform = np.eye(4)  # Create 4x4 identity matrix
        transform[:3, 3] = global_parameters[3:]
        self.translation_offset = transform

    def convert_degree_params_to_radians(self, joint_parameters, global_parameters):
        new_joint_parameters = np.array(joint_parameters)/4
        new_joint_parameters = np.deg2rad(new_joint_parameters)
        new_global_parameters = np.array(global_parameters)
        new_global_parameters[:3] = new_global_parameters[:3]/4
        new_joint_parameters[:3] = np.deg2rad(new_joint_parameters[:3])

        return list(new_joint_parameters), list(new_global_parameters)

    def get_alignment_error(self, joint_parameters, global_parameters):

        joint_parameters, global_parameters = self.convert_degree_params_to_radians(
            joint_parameters, global_parameters)

        self.apply_global_parameters(global_parameters)
        self.apply_global_transform()
        self.apply_joint_parameters(joint_parameters)
        self.apply_joint_angles()

        surface_error = self.get_surface_error()

        process_error = self.get_process_error(threshold=self.threshold)
        self.reset_spine()

        print(process_error*self.alpha + surface_error*self.beta)
        return process_error*self.alpha + surface_error*self.beta

    def objective(self, params):

        return self.get_alignment_error(params[:-6], params[-6:])

    def run_optimization(self, max_iterations=150):
        initial_params = np.zeros(((len(self.all_joints)+1)*3)+3)

        initial_params = np.zeros(((len(self.all_joints)+1)*3)+3)
        bounds = self.rotation_bounds + self.translation_bounds

        # Normalize final quaternion
        result = gp_minimize(self.objective,                  # the function to minimize
                             # the bounds on each dimension of x
                             bounds,
                             #  acq_func="EI",      # the acquisition function
                             n_calls=max_iterations,
                             # the number of evaluations of f
                             x0=list(initial_params),
                             n_random_starts=10,  # the number of random initialization points
                             noise=1e-10,
                             n_jobs=-1,
                             random_state=1234)

        final_params = result.x
        print("Ran bayesian optimization")

        print(final_params)

        return final_params, result.fun


def traverse_children(links, partition_map, upvector, forward_vector, vertices):
    head_child = None
    if len(links) == 0:
        return
    if len(links) == 1:
        # this is leaf, and has no joints
        return Vertebrae(links[0], partition_map, upvector, forward_vector, vertices)
    else:
        # can go right
        # your head will be a joint
        right_children = links[1:]

        head_child = traverse_children(
            right_children, partition_map, upvector, forward_vector, vertices)
    return Vertebrae(links[0], partition_map, upvector, forward_vector, vertices, head_child=head_child)
    # check if going right or left is valid first


def create_armature_objects(links, partition_map, spline_points, upvector, vertices):
    # links is a list of bones, defined by (tail, head position)
    # we first need to define a root node, which will be in the middle, then traverse downwards
    # joints are the head and tail of parent, and they lead to children
    # set joint parameters, figure out the fit, if good, then transform final shape

    lowest_spline_point = spline_points[0]
    p1, p2 = np.linalg.norm(
        links[0][1] - lowest_spline_point), np.linalg.norm(links[-1][0] - lowest_spline_point)

    if p1 > p2:
        links = list(reversed(links))
        links = [(x[1], x[0], x[2]) for x in links]

    centroid = np.mean(vertices, axis=0)
    forward_vector = (centroid - links[0][1])

    return traverse_children(links, partition_map, upvector, forward_vector, vertices)

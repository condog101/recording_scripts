import numpy as np
from utils import vector_to_translation_matrix, normalize_list_of_quaternion_params
import open3d as o3d
from scipy.spatial.distance import directed_hausdorff
from scipy.optimize import minimize
from scipy.optimize import Bounds
from scipy.optimize import SR1
from scipy.optimize import basinhopping
from catmull_rom import fit_catmull_rom
from utils import emd, remove_distant_points, align_centroids, filter_z_offset_points, partition
from armature_utils import get_bbox_overlap
from scipy.optimize import NonlinearConstraint


class BallJoint:
    def __init__(self, position, target_position, spline_points, x_constraint=4, y_constraint=10, z_constraint=8):
        # x axis is rotation around length of spine, z axis is towards process tip, y is towards transverse process
        # should be x:4, y:20, z:8, but increase

        self.initial_position = np.array(position)
        self.initial_target_position = np.array(target_position)
        self.position = np.array(position)
        self.target_position = target_position
        self.spline_points = spline_points
        self.x_constraint, self.y_constraint, self.z_constraint = x_constraint, y_constraint, z_constraint

        # Initialize quaternion to identity rotation [1,0,0,0]
        self.quaternion = np.array([1.0, 0.0, 0.0, 0.0])

    def update_position(self, transform):
        points = self.position.reshape(1, -1)
        homogeneous = np.hstack([points, np.ones((len(points), 1))]).T
        transformed = (transform @ homogeneous).T[:, :3][0]
        self.position[:] = transformed[:]
        target_points = self.target_position.reshape(1, -1)
        homogeneous_target = np.hstack(
            [target_points, np.ones((len(target_points), 1))]).T
        transformed_target = (transform @ homogeneous_target).T[:, :3][0]
        self.target_position[:] = transformed_target[:]

    def reset_joint_position(self):
        self.position[:] = self.initial_position[:]
        self.target_position[:] = self.initial_target_position[:]
        self.quaternion = np.array([1.0, 0.0, 0.0, 0.0])

    def set_quaternion(self, quaternion):
        self.quaternion[:] = quaternion[:]

    def set_rotation(self, quaternion):
        self.quaternion = quaternion

    def set_rotation_axis_angle(self, axis, angle):
        """Set joint rotation using axis-angle representation"""
        axis = axis / np.linalg.norm(axis)  # normalize axis
        half_angle = angle / 2.0
        self.quaternion[0] = np.cos(half_angle)
        self.quaternion[1:] = axis * np.sin(half_angle)

    def quaternion_multiply(self, q1, q2):
        """Multiply two quaternions"""
        w1, x1, y1, z1 = q1
        w2, x2, y2, z2 = q2
        return np.array([
            w1*w2 - x1*x2 - y1*y2 - z1*z2,
            w1*x2 + x1*w2 + y1*z2 - z1*y2,
            w1*y2 - x1*z2 + y1*w2 + z1*x2,
            w1*z2 + x1*y2 - y1*x2 + z1*w2
        ])

    def rotate_point(self, point):
        """Apply rotation to a point using quaternion rotation"""
        # Convert point to quaternion form (w=0)
        p_quat = np.array([0.0, *point])

        # Get quaternion conjugate
        q_conj = np.array([self.quaternion[0], *(-self.quaternion[1:])])

        # Rotate point: q * p * q'
        rotated = self.quaternion_multiply(
            self.quaternion,
            self.quaternion_multiply(p_quat, q_conj)
        )

        # Return rotated point (ignore w component)
        return rotated[1:]

    def transform_point(self, point):
        """Apply full transformation (rotation + translation) to point"""
        # First rotate
        rotated_point = self.rotate_point(point)
        # Then translate
        return rotated_point + self.position

    def constrain_rotation_matrix(self, T):
        x_limit, y_limit, z_limit = self.x_constraint, self.y_constraint, self.z_constraint

        rotation_only = T[0:3, 0:3]
        axis_rotations = decompose_rotation_matrix(rotation_only)
        conversion_factor = np.pi/180
        x_radians, y_radians, z_radians = x_limit * conversion_factor, y_limit * \
            conversion_factor, z_limit * conversion_factor

        constrained_angles = np.array([
            np.clip(axis_rotations[0], -x_radians, x_radians),
            np.clip(axis_rotations[1], -y_radians, y_radians),
            np.clip(axis_rotations[2], -z_radians, z_radians)
        ])
        cx, cy, cz = np.cos(constrained_angles)
        sx, sy, sz = np.sin(constrained_angles)

        # Build rotation matrix using XYZ order
        Rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])
        Ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
        Rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])

        R_constrained = Rz @ Ry @ Rx
        T = np.eye(4)
        T[0:3, 0:3] = R_constrained
        return T

    def to_rotation_only_matrix(self):
        q = self.quaternion / np.linalg.norm(self.quaternion)
        w, x, y, z = q

        # Create 4x4 homogeneous matrix
        matrix = np.eye(4)

        # Fill in rotation components
        matrix[0, 0] = 1 - 2*y*y - 2*z*z
        matrix[0, 1] = 2*x*y - 2*w*z
        matrix[0, 2] = 2*x*z + 2*w*y

        matrix[1, 0] = 2*x*y + 2*w*z
        matrix[1, 1] = 1 - 2*x*x - 2*z*z
        matrix[1, 2] = 2*y*z - 2*w*x

        matrix[2, 0] = 2*x*z - 2*w*y
        matrix[2, 1] = 2*y*z + 2*w*x
        matrix[2, 2] = 1 - 2*x*x - 2*y*y

        constrained_matrix = self.constrain_rotation_matrix(matrix)

        return constrained_matrix


class Vertebrae:

    def get_closest_spline_point_vector(self, joint_position):
        closest = np.argmin(
            np.sqrt(np.sum((self.spline_points - joint_position)**2, axis=1)))
        vec = self.spline_points[closest] - joint_position
        return vec / np.linalg.norm(vec)

    def get_joint_axes(self, joint: BallJoint):
        joint_position = joint.position
        toward_position = joint.target_position
        x_vector = toward_position - joint_position
        x_vector = x_vector / np.linalg.norm(x_vector)
        z_vector = find_closest_orthogonal(
            x_vector, self.get_closest_spline_point_vector(
                joint_position)
        )
        y_vector = np.cross(x_vector, z_vector)
        return x_vector, y_vector, z_vector

    def initialize_joints(self):

        if self.tail_child is not None:

            self.tail_joint = BallJoint(
                self.bone_positions[0], self.tail_child.bone_positions[0], self.spline_points)

        else:
            self.tail_joint = None

        if self.head_child is not None:
            self.head_joint = BallJoint(
                self.bone_positions[1], self.head_child.bone_positions[1], self.spline_points)
        else:
            self.head_joint = None

    def __init__(self, bone_positions, partition_map, spline_points, tail_child=None, head_child=None):
        self.bone_positions = bone_positions
        self.cluster_id = bone_positions[2]
        self.vertex_ids = partition_map[self.cluster_id]
        self.tail_child = tail_child
        self.head_child = head_child
        self.parent = None
        self.spline_points = spline_points
        self.initialize_joints()

    def set_simple_mesh_points(self, spine):

        vertebrae_centroid = np.mean(
            spine.initial_vertices[self.vertex_ids], axis=0)
        min_diff = 1000
        candidate = None
        for key, inds in spine.partition_simple.items():
            candidate_centroid = np.mean(spine.initial_vertices[inds], axis=0)
            dist = np.linalg.norm(candidate_centroid - vertebrae_centroid)
            if dist < min_diff:
                min_diff = dist
                candidate = key
        self.simple_vertex_inds = spine.partition_simple[candidate]
        if self.tail_child is not None:
            self.tail_child.set_simple_mesh_points(spine)
        if self.head_child is not None:
            self.head_child.set_simple_mesh_points(spine)

    def get_joints_and_child_joints(self):
        joints = []
        if self.tail_joint:
            joints.append(self.tail_joint)
        if self.head_joint:
            joints.append(self.head_joint)
        tail_joints = []
        if self.tail_child is not None:
            tail_joints = self.tail_child.get_joints_and_child_joints()
        head_joints = []
        if self.head_child is not None:
            head_joints = self.head_child.get_joints_and_child_joints()

        return joints + tail_joints + head_joints

    def set_parents(self, parent=None):
        # this traverses and creates links back to parent
        self.parent = parent
        if self.head_child is None and self.tail_child is None:
            return
        else:
            if self.head_child is not None:
                self.head_child.set_parents(parent=self)
            if self.tail_child is not None:
                self.tail_child.set_parents(parent=self)
            return

    def reset_joints(self):
        if self.tail_joint is not None:
            self.tail_joint.reset_joint_position()
        if self.head_joint is not None:
            self.head_joint.reset_joint_position()
        if self.head_child is not None:
            self.head_child.reset_joints()
        if self.tail_child is not None:
            self.tail_child.reset_joints()
        return

    # update joint params, propogate children position, then
    # to propogate transforms from current, I need to know, joint params,
    # qw, qx, qy, qz for each joint, as well as vertices affected by each joint, so for everything downstream, including joints bones vertices, etc
    # do translation from joint in question to coordinates origin, apply rotation, then back to
    def get_downstream_indices(self, ordinary_indices=True):

        tail_ids = []
        head_ids = []
        if self.tail_child is not None:
            tail_ids = self.tail_child.get_downstream_indices(
                ordinary_indices=ordinary_indices)
        if self.head_child is not None:
            head_ids = self.head_child.get_downstream_indices(
                ordinary_indices=ordinary_indices)
        if ordinary_indices:
            return self.vertex_ids + tail_ids + head_ids
        else:
            return self.simple_vertex_inds + tail_ids + head_ids

    @staticmethod
    def update_single_point_position(joint, transform):
        points = joint.position.reshape(1, -1)
        homogeneous = np.hstack([points, np.ones((len(points), 1))]).T
        transformed = (transform @ homogeneous).T[:, :3][0]
        joint.position = transformed

    def update_joint_positions(self, transform):
        # this should update the position of my joints, and any children I have
        if self.tail_joint is not None:
            self.tail_joint.update_position(transform)

        if self.head_joint is not None:
            self.head_joint.update_position(transform)

        if self.tail_child is not None:
            self.tail_child.update_joint_positions(transform)

        if self.head_child is not None:
            self.head_child.update_joint_positions(transform)

        return

    def propagate_joint_updates(self, spine):
        # each subject joint will have it's own affected vertices
        # first we need to update vertex and joint positions of child vertebrae affected by each joints
        # parameters, after which we then need to traverse down/up the column and
        if self.head_joint is not None:

            combined_head = self.get_sandwich_transform(self.head_joint)
            head_indices = self.head_child.get_downstream_indices()
            self.update_vertex_positions(combined_head, spine, head_indices)
            self.head_child.update_joint_positions(combined_head)
            self.head_child.propagate_joint_updates(spine)
            head_simple_indices = self.head_child.get_downstream_indices(
                ordinary_indices=False)
            self.update_vertex_positions(
                combined_head, spine, head_simple_indices, ordinary=False)

        if self.tail_joint is not None:
            combined_tail = self.get_sandwich_transform(self.tail_joint)
            tail_indices = self.tail_child.get_downstream_indices()
            self.update_vertex_positions(combined_tail, spine, tail_indices)
            self.tail_child.update_joint_positions(combined_tail)
            self.tail_child.propagate_joint_updates(spine)
            tail_simple_indices = self.tail_child.get_downstream_indices(
                ordinary_indices=False)
            self.update_vertex_positions(
                combined_tail, spine, tail_simple_indices, ordinary=False)

    @staticmethod
    def update_vertex_positions(transform, spine, indices, ordinary=True):
        if ordinary:
            spine_coords = spine.vertices[indices]
        else:
            spine_coords = spine.simple_mesh_vertices[indices]
        homogeneous = np.hstack([spine_coords, np.ones(
            (len(spine_coords), 1))]).T
        transformed = (transform @ homogeneous).T[:, :3]
        if ordinary:
            spine.vertices[indices] = transformed
        else:
            spine.simple_mesh_vertices[indices] = transformed

    def get_sandwich_transform(self, joint: BallJoint):
        # first bring the joint to origin, then align joint axes with world
        head_joint_translation = np.array(joint.position) * -1.0
        t1 = vector_to_translation_matrix(head_joint_translation)
        x1, y1, z1 = self.get_joint_axes(joint)
        rotation_alignment = align_vectors_to_axes(x1, y1, z1)
        rotation = joint.to_rotation_only_matrix()
        inv_t1 = np.linalg.inv(t1)
        inv_rotation = np.linalg.inv(rotation_alignment)
        return inv_t1 @ inv_rotation @ rotation @ rotation_alignment @ t1

    def get_submesh_voxel_error(self, vertices):

        error = 0

        if self.head_child is not None:
            error += self.head_child.get_submesh_voxel_error(vertices)
        if self.tail_child is not None:
            error += self.tail_child.get_submesh_voxel_error(vertices)

        if self.parent is not None:
            mesh1 = vertices[
                self.parent.vertex_ids]
            mesh2 = vertices[
                self.vertex_ids]
            intersection_vol, iou = get_bbox_overlap(mesh1, mesh2)
            error += iou

        return error


class Spine:

    def __init__(self, root_vertebrae, vertices_array, control_point_inds, video_curve_points, corrspondence_point_inds, scene_points, query_scene, sampled_cloud, alpha=0.5, initial_transform=None, simple_mesh=None):
        self.root_vertebrae = root_vertebrae
        self.simple_mesh = simple_mesh
        if initial_transform is not None:
            homogeneous = np.hstack([vertices_array, np.ones(
                (len(vertices_array), 1))]).T
            transformed = (initial_transform @ homogeneous).T[:, :3]

            homogeneous_simple = np.hstack([np.asarray(simple_mesh.vertices), np.ones(
                (len(np.asarray(simple_mesh.vertices)), 1))]).T
            transformed_simple = (initial_transform @
                                  homogeneous_simple).T[:, :3]
            self.initial_vertices = transformed.copy()
            self.vertices = transformed.copy()
            self.simple_mesh_vertices = transformed_simple.copy()
            self.initial_simple_mesh_vertices = transformed_simple.copy()
            self.root_vertebrae.update_joint_positions(initial_transform)
        else:
            self.simple_mesh_vertices = np.asarray(simple_mesh.vertices)
            self.initial_simple_mesh_vertices = np.asarray(
                simple_mesh.vertices)
            self.initial_vertices = vertices_array.copy()
            self.vertices = vertices_array.copy()
        self.control_point_inds = control_point_inds
        self.video_curve_points = video_curve_points
        self.correspondence_point_inds = corrspondence_point_inds
        self.scene_points = scene_points

        self.all_joints = self.get_all_joints()

        self.quaternion_bounds = [(-1.0, 1.0)
                                  for _ in range(0, len(self.all_joints) * 4)]
        self.alpha = alpha

        self.partition_simple = partition(self.simple_mesh)
        intersecting_triangles = np.asarray(
            self.simple_mesh.get_self_intersecting_triangles())
        intersecting_triangles = intersecting_triangles[0:1]
        self.intersects = len(np.unique(intersecting_triangles))
        self.set_simple_mesh_points()
        self.correspondence_distance = self.get_correspondence_distance()
        self.process_indices = self.get_tips()
        self.process_pcd = o3d.geometry.PointCloud()
        self.process_pcd.points = o3d.utility.Vector3dVector(
            self.vertices[self.process_indices])
        self.query_scene = query_scene
        self.initial_occupancy = self.process_occupancy()
        self.sampled_cloud = sampled_cloud
        self.id_transform = np.eye(4)

    def get_tips(self, tip_threshold=0.1):
        quantile = np.quantile(self.vertices[:, 2], tip_threshold)
        return np.argwhere(self.vertices[:, 2] < quantile).reshape(-1)

    def get_process_error(self, threshold=0.4):
        self.process_pcd.points = o3d.utility.Vector3dVector(
            self.vertices[self.process_indices])

        result = o3d.pipelines.registration.evaluate_registration(
            self.process_pcd, self.sampled_cloud, threshold, self.id_transform)

        correspondences = len(np.asarray(result.correspondence_set))
        min_both = min(len(self.process_indices), len(
            np.asarray(self.sampled_cloud.points)))
        return (min_both - correspondences)/min_both

    def get_overlap_error(self):
        self.simple_mesh.vertices = o3d.utility.Vector3dVector(
            self.simple_mesh_vertices)
        intersecting_triangles = np.asarray(
            self.simple_mesh.get_self_intersecting_triangles())
        intersecting_triangles = intersecting_triangles[0:1]
        intersecting_triangles_len = len(np.unique(intersecting_triangles))
        return (intersecting_triangles_len - self.intersects)

    def process_occupancy(self):
        rel_vertices = self.vertices[self.process_indices].astype(np.float32)
        return float(np.sum((self.query_scene.compute_occupancy(rel_vertices).numpy())))

    def set_simple_mesh_points(self):
        self.root_vertebrae.set_simple_mesh_points(self)

    def reset_spine(self):
        self.vertices[:, :] = self.initial_vertices[:, :]
        self.root_vertebrae.reset_joints()
        self.simple_mesh_vertices[:,
                                  :] = self.initial_simple_mesh_vertices[:, :]

    def get_all_joints(self):
        return self.root_vertebrae.get_joints_and_child_joints()

    def get_current_curve_points(self):
        rel_points = self.vertices[self.control_point_inds]
        cleaned_points = filter_z_offset_points(rel_points)

        return fit_catmull_rom(rel_points, alpha=self.alpha)

    def get_curve_similarity(self):

        current_curve_points = self.get_current_curve_points()

        # return max(directed_hausdorff(self.video_curve_points, aligned_curve)[0],
        #            directed_hausdorff(aligned_curve, self.video_curve_points)[0])
        # czero = np.zeros(len(current_curve_points))
        # # czero.fill(np.mean(current_curve_points[:, 2]))
        # czero.fill(np.mean(self.video_curve_points[:, 2]))

        # cscene = np.zeros(len(self.video_curve_points))
        # cscene.fill(np.mean(self.video_curve_points[:, 2]))
        # current_curve_points[:, 2] = cscene
        # self.video_curve_points[:, 2] = cscene

        aligned_curve = align_centroids(
            self.video_curve_points, current_curve_points)

        min_x_vid_curve = np.min(self.video_curve_points[:, 0])
        max_x_vid_curve = np.max(self.video_curve_points[:, 0])
        mask_x = (aligned_curve[:, 0] > min_x_vid_curve) & (
            aligned_curve[:, 0] < max_x_vid_curve)
        return emd(self.video_curve_points, aligned_curve[mask_x][10:-10])

    def apply_joint_parameters(self, joint_parameters):
        # joint parameters is np.array of shape NumJointsx4
        for ind, joint in enumerate(self.all_joints):
            joint.set_quaternion(joint_parameters[ind*4: (ind*4)+4])

    def apply_joint_angles(self):
        self.root_vertebrae.propagate_joint_updates(self)

    def get_correspondence_distance(self):
        mesh_points = self.vertices[self.correspondence_point_inds]
        return np.sum(np.linalg.norm(mesh_points - self.scene_points, axis=1))

    def get_correspondence_distance_error(self):
        distance = self.get_correspondence_distance()
        original_distance = self.correspondence_distance
        return np.abs(distance - original_distance)/original_distance

    def get_occupancy_error(self):
        distance = self.process_occupancy()
        original_distance = self.initial_occupancy
        return (original_distance - distance)/original_distance

    def get_alignment_error(self, joint_parameters, overlap_alpha=0.2, distance_beta=0.1, occupancy_gamma=0.5):
        self.apply_joint_parameters(joint_parameters)
        self.apply_joint_angles()
        # fitness = self.get_curve_similarity() * 0.1
        # overlap_error = self.get_overlap_error()
        overlap_error = 0
        # correspondence_distance_error = self.get_correspondence_distance_error()
        correspondence_distance_error = 0
        # occupancy_error = self.get_occupancy_error()
        process_error = self.get_process_error(threshold=0.8)
        self.reset_spine()
        # print(fitness + overlap_error*10)

        # print((1-overlap_alpha-distance_beta-occupancy_gamma)*fitness + overlap_alpha *
        #       overlap_error + correspondence_distance_error*distance_beta + occupancy_error * occupancy_gamma)
        # return (1-overlap_alpha-distance_beta-occupancy_gamma)*fitness + overlap_alpha*overlap_error + correspondence_distance_error*distance_beta + occupancy_error * occupancy_gamma
        print(process_error + (overlap_error*0.05) +
              (correspondence_distance_error*0.5))
        return process_error + (overlap_error*0.05) + \
            (correspondence_distance_error*0.5)

    def objective(self, params):
        # Normalize quaternion part before transforming
        params_normalized = normalize_list_of_quaternion_params(params)
        return self.get_alignment_error(params_normalized)

    def quaternion_constraint(self, q):
        """Unit quaternion constraint: q[0]^2 + q[1]^2 + q[2]^2 + q[3]^2 = 1"""
        quats = len(q)//4
        total = 0
        for i in range(0, quats):
            total += np.sum(q[i*4: ((i+1)*4)])
        return total - q

    def quaternion_angle_constraint(self, q, max_angle_degrees):

        quats = len(q)//4
        total = 0
        for i in range(0, quats):
            w, x, y, z = q[i*4: ((i+1)*4)]
            max_angle_rad = np.deg2rad(max_angle_degrees)

            # Constraint: |w| >= cos(x/2)
            min_w = np.cos(max_angle_rad/2)
            w_violation = min_w - abs(w)

            # Constraint: sqrt(x² + y² + z²) <= sin(x/2)
            max_xyz = np.sin(max_angle_rad/2)
            xyz_mag = np.sqrt(x*x + y*y + z*z)
            xyz_violation = xyz_mag - max_xyz

            total += np.maximum(w_violation, xyz_violation)
        return total

    def run_optimization(self, max_iterations=150, xtol=1e-8, ftol=1e-8):
        initial_params = np.zeros(len(self.all_joints)*4)
        for i in range(0, len(self.all_joints)):
            initial_params[i*4] = 1.0

        constraint = {
            'type': 'eq',
            'fun': self.quaternion_constraint
        }
        # constraint2 = NonlinearConstraint(
        #     lambda q: self.quaternion_angle_constraint(q, max_angle_degrees=5),
        #     -np.inf,
        #     0
        # )
        minimizer_kwargs = {
            "method": "SLSQP",  # or other method
            "constraints": [constraint],
            "bounds": self.quaternion_bounds
        }

        # result = minimize(
        #     self.objective,
        #     initial_params,
        #     # method='Powell',
        #     # method='L-BFGS-B',
        #     # method='trust-constr',
        #     jac="2-point", hess=SR1(),
        #     constraints=[constraint, constraint2],
        #     method='COBYLA',
        #     bounds=self.quaternion_bounds,
        #     # options={'maxiter': max_iterations,
        #     #          'disp': True, 'ftol': ftol, 'xtol': xtol}
        #     options={'maxiter': max_iterations,
        #              'disp': True}
        # )
        result = basinhopping(self.objective, initial_params,
                              minimizer_kwargs=minimizer_kwargs,
                              niter=20,  # number of basin hopping iterations
                              T=1.0,      # temperature parameter for acceptance
                              stepsize=0.5  # initial step size for perturbation
                              )

        # Normalize final quaternion
        final_params = result.x
        print(f"optimizer succeeded: {result.success}")
        final_params = normalize_list_of_quaternion_params(final_params)
        print(final_params)

        return final_params, result.fun


def traverse_children(links, parent_index, partition_map, spline_points):
    tail_child = None
    head_child = None
    if len(links) == 1:
        # this is leaf, and has no joints
        return Vertebrae(links[0], partition_map, spline_points)
    if parent_index < len(links) - 1:
        # can go right
        # your head will be a joint
        right_children = links[parent_index + 1:]
        ind_r = 0
        head_child = traverse_children(
            right_children, ind_r, partition_map, spline_points)
    if parent_index > 0:
        # can go left
        # your tail will be a joint
        left_children = links[: parent_index]
        ind_l = parent_index - 1
        tail_child = traverse_children(
            left_children, ind_l, partition_map, spline_points)
    return Vertebrae(links[parent_index], partition_map, spline_points, tail_child=tail_child, head_child=head_child)
    # check if going right or left is valid first


def create_armature_objects(links, partition_map, spline_points):
    # links is a list of bones, defined by (tail, head position)
    # we first need to define a route node, which will be in the middle, then traverse downwards
    # joints are the head and tail of parent, and they lead to children
    # set joint parameters, figure out the fit, if good, then transform final shape
    parent_index = len(links)//2

    return traverse_children(links, parent_index, partition_map, spline_points)


def find_closest_orthogonal(basis_vector, target_vector):
    """
    Find vector orthogonal to basis_vector that's closest to target_vector

    Args:
        basis_vector: First unit vector that defines primary axis
        target_vector: Second unit vector we want to get close to

    Returns:
        orthogonal unit vector closest to target_vector
    """
    # Project target onto basis
    projection = np.dot(target_vector, basis_vector) * basis_vector

    # Subtract projection to get orthogonal component
    orthogonal = target_vector - projection

    # Normalize result
    return orthogonal / np.linalg.norm(orthogonal)


def align_vectors_to_axes(vec_for_x, vec_for_y, vec_for_z):
    """
    Creates a rotation matrix that aligns three orthogonal vectors with the coordinate axes.

    Args:
        vec_for_x: 3D vector to be aligned with x-axis [1,0,0]
        vec_for_y: 3D vector to be aligned with y-axis [0,1,0]
        vec_for_z: 3D vector to be aligned with z-axis [0,0,1]

    Returns:
        rotation_matrix: 4x4 transformation matrix
    """

    # Create matrix from input vectors (each vector is a column)
    source_frame = np.column_stack([vec_for_x, vec_for_y, vec_for_z])

    # Create matrix for target coordinate frame
    target_frame = np.eye(3)  # Identity matrix represents standard basis

    # The rotation matrix is R = target_frame @ source_frame^T
    rotation_matrix = target_frame @ np.linalg.inv(source_frame)

    transform = np.eye(4)  # Create 4x4 identity matrix
    transform[:3, :3] = rotation_matrix

    return transform


def decompose_rotation_matrix(R):
    """
    Decompose a 3x3 rotation matrix into Euler angles (in radians)
    Using Tait-Bryan angles with order XYZ
    """
    # Handle numerical errors that might make values slightly out of [-1,1]
    sy = np.sqrt(R[0, 0] * R[0, 0] + R[1, 0] * R[1, 0])

    if sy > 1e-6:
        x = np.arctan2(R[2, 1], R[2, 2])
        y = np.arctan2(-R[2, 0], sy)
        z = np.arctan2(R[1, 0], R[0, 0])
    else:
        x = np.arctan2(-R[1, 2], R[1, 1])
        y = np.arctan2(-R[2, 0], sy)
        z = 0

    return np.array([x, y, z])

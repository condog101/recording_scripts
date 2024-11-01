import numpy as np
from utils import vector_to_translation_matrix, normalize_list_of_quaternion_params
from scipy.interpolate import Rbf
from scipy.spatial.distance import directed_hausdorff
from scipy.optimize import minimize
from catmull_rom import fit_catmull_rom
from utils import emd


class BallJoint:
    def __init__(self, position):
        self.initial_position = np.array(position)
        self.position = np.array(position)
        # Initialize quaternion to identity rotation [1,0,0,0]
        self.quaternion = np.array([1.0, 0.0, 0.0, 0.0])

    def update_position(self, transform):
        points = self.position.reshape(1, -1)
        homogeneous = np.hstack([points, np.ones((len(points), 1))]).T
        transformed = (transform @ homogeneous).T[:, :3][0]
        self.position[:] = transformed[:]

    def reset_joint_position(self):
        self.position[:] = self.initial_position[:]
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

    def to_matrix(self):
        """Convert to 4x4 transformation matrix"""
        w, x, y, z = self.quaternion

        matrix = np.array([
            [1-2*y*y-2*z*z,  2*x*y-2*w*z,    2*x*z+2*w*y,    self.position[0]],
            [2*x*y+2*w*z,    1-2*x*x-2*z*z,  2*y*z-2*w*x,    self.position[1]],
            [2*x*z-2*w*y,    2*y*z+2*w*x,    1-2*x*x-2*y*y,  self.position[2]],
            [0,              0,              0,              1]
        ])

        return matrix

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

        return matrix


class Vertebrae:

    def initialize_joints(self):

        if self.tail_child is not None:

            self.tail_joint = BallJoint(self.bone_positions[0])

        else:
            self.tail_joint = None

        if self.head_child is not None:

            self.head_joint = BallJoint(self.bone_positions[1])
        else:
            self.head_joint = None

    def __init__(self, bone_positions, partition_map, tail_child=None, head_child=None):
        self.bone_positions = bone_positions
        self.cluster_id = bone_positions[2]
        self.vertex_ids = partition_map[self.cluster_id]
        self.tail_child = tail_child
        self.head_child = head_child
        self.parent = None
        self.initialize_joints()

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
    def get_downstream_indices(self):
        tail_ids = []
        head_ids = []
        if self.tail_child is not None:
            tail_ids = self.tail_child.get_downstream_indices()
        if self.head_child is not None:
            head_ids = self.head_child.get_downstream_indices()
        return self.vertex_ids + tail_ids + head_ids

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

        if self.tail_joint is not None:
            combined_tail = self.get_sandwich_transform(self.tail_joint)
            tail_indices = self.tail_child.get_downstream_indices()
            self.update_vertex_positions(combined_tail, spine, tail_indices)
            self.tail_child.update_joint_positions(combined_tail)
            self.tail_child.propagate_joint_updates(spine)

    @staticmethod
    def update_vertex_positions(transform, spine, indices):
        spine_coords = spine.vertices[indices]
        homogeneous = np.hstack([spine_coords, np.ones(
            (len(spine_coords), 1))]).T
        transformed = (transform @ homogeneous).T[:, :3]
        spine.vertices[indices] = transformed

    @staticmethod
    def get_sandwich_transform(joint):
        head_joint_translation = np.array(joint.position) * -1.0
        t1 = vector_to_translation_matrix(head_joint_translation)
        rotation = joint.to_rotation_only_matrix()
        inv_t1 = np.linalg.inv(t1)
        return inv_t1 @ rotation @ t1


class Spine:

    def __init__(self, root_vertebrae, vertices_array, control_point_inds, video_curve_points, initial_transform=None, grid_sample_points=500):
        self.root_vertebrae = root_vertebrae

        if initial_transform is not None:
            homogeneous = np.hstack([vertices_array, np.ones(
                (len(vertices_array), 1))]).T
            transformed = (initial_transform @ homogeneous).T[:, :3]
            self.initial_vertices = transformed.copy()
            self.vertices = transformed.copy()
            self.root_vertebrae.update_joint_positions(initial_transform)
        else:
            self.initial_vertices = vertices_array.copy()
            self.vertices = vertices_array.copy()
        self.control_point_inds = control_point_inds
        self.video_curve_points = video_curve_points

        self.all_joints = self.get_all_joints()

        self.quaternion_bounds = [(-1.0, 1.0)
                                  for _ in range(0, len(self.all_joints) * 4)]

    def reset_spine(self):
        self.vertices[:, :] = self.initial_vertices[:, :]
        self.root_vertebrae.reset_joints()

    def get_all_joints(self):
        return self.root_vertebrae.get_joints_and_child_joints()

    def get_current_curve_points(self):
        rel_points = self.vertices[self.control_point_inds]
        return fit_catmull_rom(rel_points)

    def get_curve_similarity(self):
        current_curve_points = self.get_current_curve_points()

        # import open3d as o3d
        # pcd1 = o3d.geometry.PointCloud()
        # pcd1.points = o3d.utility.Vector3dVector(current_curve_points)
        # pcd2 = o3d.geometry.PointCloud()
        # pcd2.points = o3d.utility.Vector3dVector(self.video_points)
        # pcd1.colors = o3d.utility.Vector3dVector(
        #     np.array([[1, 0, 0] for _ in current_curve_points]))
        # pcd2.colors = o3d.utility.Vector3dVector(
        #     np.array([[0, 1, 0] for _ in self.video_points]))
        # o3d.visualization.draw_geometries([pcd1, pcd2])

        # return max(directed_hausdorff(self.video_curve_points, current_curve_points)[0],
        #            directed_hausdorff(current_curve_points, self.video_curve_points)[0])
        return emd(self.video_curve_points, current_curve_points)

    def apply_joint_parameters(self, joint_parameters):
        # joint parameters is np.array of shape NumJointsx4
        for ind, joint in enumerate(self.all_joints):
            joint.set_quaternion(joint_parameters[ind*4: (ind*4)+4])

    def apply_joint_angles(self):
        self.root_vertebrae.propagate_joint_updates(self)

    def get_alignment_error(self, joint_parameters):
        self.apply_joint_parameters(joint_parameters)
        self.apply_joint_angles()
        fitness = self.get_curve_similarity()
        self.reset_spine()
        print(fitness)
        return fitness

    def objective(self, params):
        # Normalize quaternion part before transforming
        params_normalized = normalize_list_of_quaternion_params(params)
        return self.get_alignment_error(params_normalized)

    def run_optimization(self, max_iterations=500, xtol=1e-8, ftol=1e-8):
        initial_params = np.zeros(len(self.all_joints)*4)
        for i in range(0, len(self.all_joints)):
            initial_params[i*4] = 1.0

        result = minimize(
            self.objective,
            initial_params,
            # method='Powell',
            method='L-BFGS-B',
            bounds=self.quaternion_bounds,

            options={'maxiter': max_iterations, 'disp': True, 'xtol': xtol,
                     'ftol': ftol, 'return_all': True}
        )

        # Normalize final quaternion
        final_params = result.x
        print(f"optimizer succeeded: {result.success}")
        final_params = normalize_list_of_quaternion_params(final_params)

        return final_params, result.fun


def traverse_children(links, parent_index, partition_map):
    tail_child = None
    head_child = None
    if len(links) == 1:
        # this is leaf, and has no joints
        return Vertebrae(links[0], partition_map)
    if parent_index < len(links) - 1:
        # can go right
        # your head will be a joint
        right_children = links[parent_index + 1:]
        ind_r = 0
        head_child = traverse_children(right_children, ind_r, partition_map)
    if parent_index > 0:
        # can go left
        # your tail will be a joint
        left_children = links[: parent_index]
        ind_l = parent_index - 1
        tail_child = traverse_children(left_children, ind_l, partition_map)
    return Vertebrae(links[parent_index], partition_map, tail_child=tail_child, head_child=head_child)
    # check if going right or left is valid first


def create_armature_objects(links, partition_map):
    # links is a list of bones, defined by (tail, head position)
    # we first need to define a route node, which will be in the middle, then traverse downwards
    # joints are the head and tail of parent, and they lead to children
    # set joint parameters, figure out the fit, if good, then transform final shape
    parent_index = len(links)//2

    return traverse_children(links, parent_index, partition_map)

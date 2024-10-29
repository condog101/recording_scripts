import numpy as np


class BallJoint:
    def __init__(self, position):
        self.position = np.array(position)
        # Initialize quaternion to identity rotation [1,0,0,0]
        self.quaternion = np.array([1.0, 0.0, 0.0, 0.0])

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

    def __init__(self, bone_positions, tail_child=None, head_child=None):
        self.bone_positions = bone_positions
        self.cluster_id = bone_positions[2]
        self.tail_child = tail_child
        self.head_child = head_child
        self.parent = None
        self.initialize_joints()

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

    # update joint params, propogate children position, then


def traverse_children(links, parent_index):
    tail_child = None
    head_child = None
    if len(links) == 1:
        # this is leaf, and has no joints
        return Vertebrae(links[0])
    if parent_index < len(links) - 1:
        # can go right
        # your head will be a joint
        right_children = links[parent_index + 1:]
        ind_r = 0
        head_child = traverse_children(right_children, ind_r)
    if parent_index > 0:
        # can go left
        # your tail will be a joint
        left_children = links[: parent_index]
        ind_l = parent_index - 1
        tail_child = traverse_children(left_children, ind_l)
    return Vertebrae(links[parent_index], tail_child=tail_child, head_child=head_child)
    # check if going right or left is valid first


def create_armature_objects(links):
    # links is a list of bones, defined by (tail, head position)
    # we first need to define a route node, which will be in the middle, then traverse downwards
    # joints are the head and tail of parent, and they lead to children
    parent_index = len(links)//2

    return traverse_children(links, parent_index)

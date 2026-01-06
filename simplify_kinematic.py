import numpy as np
from segment_helpers.utilities import load_meshes, load_meshes_o3d
import os
import pickle as pkl
import trimesh
import copy
from trimesh.collision import CollisionManager
from sklearn.cluster import DBSCAN


def compute_volumetric_centroid(mesh):
    """
    Compute the centroid of a mesh using trimesh's center_mass.

    This uses the mesh's moment of inertia to compute a true volumetric 
    center of mass, which is much faster than sampling and accounts for
    the full volume distribution.

    Args:
        mesh: trimesh.Trimesh object

    Returns:
        np.ndarray: (3,) centroid coordinates
    """
    return mesh.center_mass


def find_planar_disc_faces(mesh, spine_axis=None, normal_threshold=0.85, min_faces=10):
    """
    Find planar faces on a vertebra mesh that likely correspond to disc contact surfaces.

    These are the relatively flat superior (top) and inferior (bottom) endplates
    where the intervertebral disc contacts the vertebral body.

    Args:
        mesh: trimesh.Trimesh object
        spine_axis: (3,) vector indicating spine direction (up). If None, uses [0,0,1]
        normal_threshold: minimum dot product with spine axis to consider a face as disc-facing
        min_faces: minimum number of faces to consider a valid planar region

    Returns:
        tuple: (top_face_indices, bottom_face_indices) - arrays of face indices
    """
    if spine_axis is None:
        spine_axis = np.array([0, 0, 1])
    spine_axis = spine_axis / np.linalg.norm(spine_axis)

    # Get face normals and centroids
    face_normals = mesh.face_normals
    face_centroids = mesh.triangles_center

    # Find faces whose normals are roughly aligned with spine axis (top faces)
    # or anti-aligned (bottom faces)
    dot_products = np.dot(face_normals, spine_axis)

    # Top faces: normals pointing in same direction as spine axis
    top_candidates = np.where(dot_products > normal_threshold)[0]

    # Bottom faces: normals pointing opposite to spine axis
    bottom_candidates = np.where(dot_products < -normal_threshold)[0]

    # For each set of candidates, cluster by position to find the main planar region
    # (vertebrae may have multiple small flat regions, we want the main endplates)

    def cluster_and_select_largest(face_indices, centroids):
        """Cluster faces by position and return the largest cluster."""
        if len(face_indices) < min_faces:
            return np.array([], dtype=int)

        positions = centroids[face_indices]

        # Use DBSCAN to cluster faces by position
        # eps is the max distance between samples in a cluster
        mesh_scale = np.linalg.norm(mesh.bounds[1] - mesh.bounds[0])
        eps = mesh_scale * 0.1  # 10% of mesh size

        clustering = DBSCAN(eps=eps, min_samples=3).fit(positions)
        labels = clustering.labels_

        # Find the largest cluster (excluding noise label -1)
        unique_labels = set(labels) - {-1}
        if not unique_labels:
            return face_indices  # Return all if no clear clusters

        largest_cluster_label = max(
            unique_labels, key=lambda l: np.sum(labels == l))
        cluster_mask = labels == largest_cluster_label

        return face_indices[cluster_mask]

    top_faces = cluster_and_select_largest(top_candidates, face_centroids)
    bottom_faces = cluster_and_select_largest(
        bottom_candidates, face_centroids)

    return top_faces, bottom_faces


def find_disc_faces_for_vertebrae_chain(meshes, joint_centers=None):
    """
    Find disc contact faces for a chain of vertebrae.

    Args:
        meshes: List of trimesh.Trimesh objects (ordered along spine)
        joint_centers: Optional (N-1, 3) array of joint positions to determine spine axis

    Returns:
        list of tuples: [(top_faces, bottom_faces), ...] for each vertebra
    """
    n_verts = len(meshes)

    # Determine spine axis from centroids if joint_centers not provided
    if joint_centers is None:
        centroids = np.array([m.center_mass for m in meshes])
        if len(centroids) > 1:
            spine_axis = centroids[-1] - centroids[0]
            spine_axis = spine_axis / np.linalg.norm(spine_axis)
        else:
            spine_axis = np.array([0, 0, 1])
    else:
        if len(joint_centers) > 1:
            spine_axis = joint_centers[-1] - joint_centers[0]
            spine_axis = spine_axis / np.linalg.norm(spine_axis)
        else:
            spine_axis = np.array([0, 0, 1])

    all_disc_faces = []
    for mesh in meshes:
        top_faces, bottom_faces = find_planar_disc_faces(mesh, spine_axis)
        all_disc_faces.append((top_faces, bottom_faces))

    return all_disc_faces


def compute_disc_face_centroid(mesh, face_indices):
    """
    Compute the centroid of a set of faces on a mesh.

    Args:
        mesh: trimesh.Trimesh object
        face_indices: array of face indices

    Returns:
        np.ndarray: (3,) centroid of the faces, or None if no faces
    """
    if face_indices is None or len(face_indices) == 0:
        return None

    # Get face centroids and areas for weighted average
    face_centroids = mesh.triangles_center[face_indices]
    face_areas = mesh.area_faces[face_indices]

    # Weighted centroid by face area
    total_area = np.sum(face_areas)
    if total_area == 0:
        return np.mean(face_centroids, axis=0)

    weighted_centroid = np.sum(
        face_centroids * face_areas[:, np.newaxis], axis=0) / total_area
    return weighted_centroid


def compute_disc_face_normal(mesh, face_indices):
    """
    Compute the average normal of a set of faces on a mesh.

    Args:
        mesh: trimesh.Trimesh object
        face_indices: array of face indices

    Returns:
        np.ndarray: (3,) normalized average normal, or None if no faces
    """
    if face_indices is None or len(face_indices) == 0:
        return None

    # Get face normals and areas for weighted average
    face_normals = mesh.face_normals[face_indices]
    face_areas = mesh.area_faces[face_indices]

    # Weighted average normal by face area
    total_area = np.sum(face_areas)
    if total_area == 0:
        avg_normal = np.mean(face_normals, axis=0)
    else:
        avg_normal = np.sum(
            face_normals * face_areas[:, np.newaxis], axis=0) / total_area

    # Normalize
    norm = np.linalg.norm(avg_normal)
    if norm > 0:
        avg_normal = avg_normal / norm

    return avg_normal


def compute_joint_from_disc_faces(mesh_above, mesh_below, disc_faces_above, disc_faces_below, x_offset=-3.0):
    """
    Compute joint center and axis using disc face geometry.

    The joint center is the midpoint between the centroids of the adjacent disc faces,
    with an optional offset in the local x direction.
    The joint axis (spine direction) is derived from the disc face normals.

    Args:
        mesh_above: trimesh.Trimesh of the vertebra above the joint
        mesh_below: trimesh.Trimesh of the vertebra below the joint
        disc_faces_above: (top_faces, bottom_faces) tuple for upper vertebra
        disc_faces_below: (top_faces, bottom_faces) tuple for lower vertebra
        x_offset: offset in the local x direction (negative = anterior/front direction)

    Returns:
        tuple: (joint_center, spine_axis) or (None, None) if insufficient data
    """
    # The joint is between:
    # - bottom faces of upper vertebra (inferior endplate)
    # - top faces of lower vertebra (superior endplate)
    # bottom/inferior of upper vertebra
    bottom_faces_upper = disc_faces_above[1]
    top_faces_lower = disc_faces_below[0]      # top/superior of lower vertebra

    centroid_upper = compute_disc_face_centroid(mesh_above, bottom_faces_upper)
    centroid_lower = compute_disc_face_centroid(mesh_below, top_faces_lower)

    # Fallback to mesh centroids if disc faces not found
    if centroid_upper is None:
        centroid_upper = mesh_above.center_mass
    if centroid_lower is None:
        centroid_lower = mesh_below.center_mass

    # Joint center is midpoint between disc face centroids
    joint_center = (centroid_upper + centroid_lower) / 2.0

    # Compute spine axis from disc face normals
    normal_upper = compute_disc_face_normal(mesh_above, bottom_faces_upper)
    normal_lower = compute_disc_face_normal(mesh_below, top_faces_lower)

    if normal_upper is not None and normal_lower is not None:
        # Average the normals (they should point in opposite directions along spine)
        # The bottom face normal points down, top face normal points up
        # We want the spine axis pointing from lower to upper vertebra
        spine_axis = (-normal_upper + normal_lower) / 2.0
        norm = np.linalg.norm(spine_axis)
        if norm > 0:
            spine_axis = spine_axis / norm
        else:
            spine_axis = centroid_upper - centroid_lower
            spine_axis = spine_axis / np.linalg.norm(spine_axis)
    else:
        # Fallback: use direction between centroids
        spine_axis = centroid_upper - centroid_lower
        norm = np.linalg.norm(spine_axis)
        if norm > 0:
            spine_axis = spine_axis / norm
        else:
            spine_axis = np.array([0, 0, 1])

    # Compute local x-axis (perpendicular to spine axis)
    # This will be used to apply the x_offset
    ref_vector = np.array([1, 0, 0])
    if abs(np.dot(spine_axis, ref_vector)) > 0.9:
        ref_vector = np.array([0, 1, 0])

    x_axis = np.cross(spine_axis, ref_vector)
    x_axis = x_axis / np.linalg.norm(x_axis)

    # Apply x offset to joint center
    joint_center = joint_center + x_offset * x_axis

    return joint_center, spine_axis


def compute_joints_from_disc_faces(meshes, disc_faces_list):
    """
    Compute joint centers and axes for a vertebrae chain using disc face geometry.

    Args:
        meshes: List of trimesh.Trimesh objects (ordered along spine)
        disc_faces_list: List of (top_faces, bottom_faces) tuples for each vertebra

    Returns:
        tuple: (joint_centers, joint_axes)
            - joint_centers: (N-1, 3) array of joint positions
            - joint_axes: (N-1, 3, 3) array of axes for each joint [x, y, z]
    """
    n_joints = len(meshes) - 1
    joint_centers = []
    spine_axes = []

    for i in range(n_joints):
        center, spine_axis = compute_joint_from_disc_faces(
            meshes[i], meshes[i + 1],
            disc_faces_list[i], disc_faces_list[i + 1]
        )
        joint_centers.append(center)
        spine_axes.append(spine_axis)

    joint_centers = np.array(joint_centers)

    # Now compute full coordinate frames for each joint
    all_axes = []
    for i, spine_axis in enumerate(spine_axes):
        # Z-axis: along the spine direction
        z_axis = spine_axis

        # X-axis: perpendicular to z, in a consistent direction
        ref_vector = np.array([1, 0, 0])
        if abs(np.dot(z_axis, ref_vector)) > 0.9:
            ref_vector = np.array([0, 1, 0])

        x_axis = np.cross(z_axis, ref_vector)
        x_axis = x_axis / np.linalg.norm(x_axis)

        # Y-axis: perpendicular to both x and z
        y_axis = np.cross(z_axis, x_axis)
        y_axis = y_axis / np.linalg.norm(y_axis)

        all_axes.append([x_axis, y_axis, z_axis])

    return joint_centers, np.array(all_axes)


def compute_joint_centroid_between_vertebrae(mesh_above, mesh_below):
    """
    Compute the joint center between two adjacent vertebrae.

    The joint center is estimated as the midpoint between the volumetric
    centroids of the two vertebrae.

    Args:
        mesh_above: trimesh.Trimesh of the vertebra above the joint
        mesh_below: trimesh.Trimesh of the vertebra below the joint

    Returns:
        np.ndarray: (3,) joint center coordinates
    """
    centroid_above = compute_volumetric_centroid(mesh_above)
    centroid_below = compute_volumetric_centroid(mesh_below)

    # Joint center is midpoint between adjacent vertebrae centroids
    joint_center = (centroid_above + centroid_below) / 2.0

    return joint_center


def compute_all_joint_centroids(meshes):
    """
    Compute joint centroids for a chain of vertebrae meshes.

    Args:
        meshes: List of trimesh.Trimesh objects in order (top to bottom or vice versa)

    Returns:
        np.ndarray: (N-1, 3) array of joint center positions, where N is number of vertebrae
    """
    joint_centers = []

    for i in range(len(meshes) - 1):
        joint_center = compute_joint_centroid_between_vertebrae(
            meshes[i], meshes[i + 1]
        )
        joint_centers.append(joint_center)

    return np.array(joint_centers)


def compute_joint_axes_from_centroids(joint_centers, up_vector=None):
    """
    Compute local coordinate axes for each joint based on the spine direction.

    Args:
        joint_centers: (N-1, 3) array of joint center positions
        up_vector: Optional (3,) vector indicating the 'up' direction of the spine.
                   If None, uses the direction from first to last joint.

    Returns:
        np.ndarray: (N-1, 3, 3) array of axes for each joint
                    Each joint has 3 axes: [x_axis, y_axis, z_axis]
    """
    n_joints = len(joint_centers)
    all_axes = []

    # Determine spine direction (z-axis will align with spine)
    if up_vector is None:
        if n_joints > 1:
            up_vector = joint_centers[-1] - joint_centers[0]
        else:
            up_vector = np.array([0, 0, 1])

    up_vector = up_vector / np.linalg.norm(up_vector)

    for i in range(n_joints):
        # Z-axis: along the spine direction
        if i < n_joints - 1:
            z_axis = joint_centers[i + 1] - joint_centers[i]
        else:
            z_axis = joint_centers[i] - joint_centers[i - 1]
        z_axis = z_axis / np.linalg.norm(z_axis)

        # X-axis: perpendicular to z, in a consistent direction
        # Use a reference vector to establish x
        ref_vector = np.array([1, 0, 0])
        if abs(np.dot(z_axis, ref_vector)) > 0.9:
            ref_vector = np.array([0, 1, 0])

        x_axis = np.cross(z_axis, ref_vector)
        x_axis = x_axis / np.linalg.norm(x_axis)

        # Y-axis: perpendicular to both x and z
        y_axis = np.cross(z_axis, x_axis)
        y_axis = y_axis / np.linalg.norm(y_axis)

        all_axes.append([x_axis, y_axis, z_axis])

    return np.array(all_axes)


class Joint:

    @staticmethod
    def rodrigues(axis, theta):
        """
        axis : (3,) unit vector
        theta: rotation angle (rad)
        returns 3×3 rotation matrix exp(  [axis]× * θ )
        """
        a = axis / np.linalg.norm(axis)
        K = np.array([[0, -a[2],  a[1]],
                      [a[2],     0, -a[0]],
                      [-a[1],  a[0],     0]])
        return np.eye(3) + np.sin(theta)*K + (1 - np.cos(theta))*K@K

    def __init__(self, center, axes, parent):
        self.parent = parent
        self.transform = np.eye(4)
        self.T_joint = np.eye(4)
        self.T_joint[:3, :3] = np.column_stack((axes[0], axes[1], axes[2]))
        self.T_joint[:3, 3] = center
        # Store original values for reset capability
        self.original_center = center.copy()
        self.original_axes = [ax.copy() for ax in axes]
        self.original_T_joint = self.T_joint.copy()
        self.fparams = np.shape(6)
        self.local_axes = [np.array([1, 0, 0]), np.array(
            [0, 1, 0]), np.array([0, 0, 1])]
        self.center = center.copy()
        self.axes = [ax.copy() for ax in axes]

    def update_transform(self, new_transform):
        self.transform = self.transform @ new_transform

    def rotate_along_local_axis(self, local_axis, theta):

        theta_rad = np.radians(theta)
        rot_axis = self.local_axes[local_axis]

        R_loc = Joint.rodrigues(rot_axis, theta_rad)
        T_loc = np.eye(4)
        T_loc[:3, :3] = R_loc

        return self.T_joint @ T_loc @ np.linalg.inv(self.T_joint)

    def translate_along_local_axis(self, local_axis, distance):

        trans_axis = self.local_axes[local_axis]

        t_local = distance * trans_axis
        T_loc = np.eye(4)
        T_loc[:3, 3] = t_local

        return self.T_joint @ T_loc @ np.linalg.inv(self.T_joint)

    def get_fparam_transform(self, fparams):
        # fparams is an array of 6 values describing axis angle rotations and translations
        # based on the UI there should be only 1 non zero value, but just in case iterate through all
        transformation_matrices = []
        for ind, param in enumerate(fparams):
            # if ind < 3 is a rotation
            # else translation
            if ind < 3:
                transformation_matrices.append(
                    self.rotate_along_local_axis(ind, param))
            else:
                transformation_matrices.append(
                    self.translate_along_local_axis(ind - 3, param))

        acc = transformation_matrices[0]
        [acc := acc @ x for x in transformation_matrices[1:]]
        return acc

    def propagate_fparam_transform(self, collision_manager, fparams, skip_collision_check=False):
        transform = self.get_fparam_transform(fparams)

        if skip_collision_check:
            # Directly apply transform without collision checking
            self.parent.propagate_transform(transform)
            # Keep collision manager in sync with actual transforms
            self.parent.sync_collision_transforms(collision_manager, transform)
            return

        # First, test the transform in the collision manager without applying to actual meshes
        # Collect current transforms to restore if needed
        candidate_transforms = self.parent.compute_candidate_transforms(
            transform)

        # Apply candidate transforms to collision manager to test
        self.parent.apply_candidate_to_collision_manager(
            collision_manager, candidate_transforms)

        # Check if this would cause too many collisions
        if collision_manager.is_more_collisions():
            # Revert collision manager to current actual transforms
            self.parent.revert_collision_manager(collision_manager)
            # Don't apply the transform to actual meshes
            return

        # Collision check passed - apply transform to actual meshes
        self.parent.propagate_transform(transform)


class Vertebra:

    def __init__(self, name, mesh, mesh_o3d, disc_faces_bottom=None, disc_faces_top=None):

        self.name = name
        self.transform = np.eye(4)
        self.mesh = mesh
        self.order = None
        self.proximity_query = trimesh.proximity.ProximityQuery(self.mesh)
        # o3d meshes are needed for visualization function, these meshes will be transformed in the adjustment
        # process
        self.mesh_o3d = mesh_o3d
        self.joint = None
        self.disc_faces_bottom = disc_faces_bottom
        self.disc_faces_top = disc_faces_top
        self.parent = None
        self.child = None

    def initialize_joint(self, center, axes):
        self.joint = Joint(center, axes, self)

    def update_transform(self, new_transform):
        # reset o3d to original pose, then move to new pose position

        # if this transform will introduce more collisions prevent movement in chain

        self.mesh_o3d.transform(np.linalg.inv(self.transform))
        self.transform = self.transform @ new_transform

        self.mesh_o3d.transform(self.transform)

        if self.joint is not None:
            self.joint.update_transform(self.transform)

    def propagate_transform(self, new_transform):
        if self.child is not None:
            self.child.update_transform(new_transform)
            self.child.propagate_transform(new_transform)

    def compute_candidate_transforms(self, new_transform):
        """
        Compute what the transforms would be after applying new_transform,
        without actually applying them.

        Returns:
            dict: mapping of vertebra name to candidate transform matrix
        """
        candidates = {}
        node = self.child
        while node is not None:
            candidate_transform = node.transform @ new_transform
            candidates[node.name] = candidate_transform
            node = node.child
        return candidates

    def apply_candidate_to_collision_manager(self, collision_manager, candidate_transforms):
        """
        Apply candidate transforms to the collision manager for testing.

        Args:
            collision_manager: CollisionManagerSpine instance
            candidate_transforms: dict mapping vertebra names to transform matrices
        """
        for name, transform in candidate_transforms.items():
            collision_manager.set_transform(name, transform)

    def revert_collision_manager(self, collision_manager):
        """
        Revert collision manager transforms to match actual current vertebra transforms.
        """
        node = self.child
        while node is not None:
            collision_manager.set_transform(node.name, node.transform)
            node = node.child

    def sync_collision_transforms(self, collision_manager, new_transform):
        """
        Sync collision manager with actual transforms after a transform is applied.
        Call this after propagate_transform when skipping collision check.
        """
        node = self.child
        while node is not None:
            collision_manager.set_transform(node.name, node.transform)
            node = node.child


class CollisionManagerSpine:

    def get_collision_depth(self):
        """Get total penetration depth (sum of all contact depths)."""
        _, _, collisions = self.manager.in_collision_internal(
            return_names=True, return_data=True)
        if not collisions:
            return 0.0
        return sum(c.depth for c in collisions)

    def get_collision_count(self):
        """Get number of collision contacts."""
        _, _, collisions = self.manager.in_collision_internal(
            return_names=True, return_data=True)
        return len(collisions)

    def set_initial_collision_depth(self):
        self.initial_depth = self.get_collision_depth()
        self.initial_count = self.get_collision_count()
        print(
            f"Initial collision state: depth={self.initial_depth:.2f}, contacts={self.initial_count}")

    def __init__(self):
        self.manager = CollisionManager()
        self.initial_depth = 0.0
        self.initial_count = 0

    def is_more_collisions(self, depth_tolerance=0.05, count_tolerance=1.05):
        """
        Check if current collisions exceed acceptable threshold.

        Args:
            depth_tolerance: allowed fractional increase in total penetration depth (0.05 = 5%)
            count_tolerance: allowed fractional increase in contact count (1.05 = 5%)

        Returns:
            bool: True if collisions exceed threshold
        """
        _, _, collisions = self.manager.in_collision_internal(
            return_names=True, return_data=True)

        if not collisions:
            return False

        current_depth = sum(c.depth for c in collisions)
        current_count = len(collisions)

        # Use penetration depth as primary metric
        # Allow small increase from initial state
        depth_threshold = self.initial_depth * \
            (1.0 + depth_tolerance) + 0.1  # +0.1 for small absolute tolerance

        if current_depth > depth_threshold:
            return True

        return False

    def add_object(self, name, mesh):
        self.manager.add_object(name, mesh)

    def set_transform(self, name, transform):
        self.manager.set_transform(name, transform)


class Spine:

    @staticmethod
    def reallocate_face_indices_per_vertebra(face_indices):
        # new pairings is in order of bottom top in tuple
        new_pairings = []
        for ind in range(0, len(face_indices) + 1):
            if ind == 0:
                new_pairings.append((None, face_indices[0][0]))
            elif ind == len(face_indices):
                new_pairings.append((face_indices[ind - 1][1], None))
            else:
                new_pairings.append(
                    (face_indices[ind - 1][1], face_indices[ind][0]))
        return new_pairings

    def load_and_set_vertebrae(self, vertebrae_names, directory_path, compute_centroids_if_missing=True, use_disc_faces_for_joints=True):
        meshes = load_meshes(directory_path, vertebrae_names)
        open3d_meshes = load_meshes_o3d(directory_path, vertebrae_names)

        # File paths for cached data
        centroids_path = os.path.join(directory_path, "centroids.npy")
        axes_path = os.path.join(directory_path, "axes.npy")
        face_indices_path = os.path.join(directory_path, "face_indices.pkl")
        disc_faces_path = os.path.join(directory_path, "disc_faces.pkl")

        # First, compute or load disc faces (needed for both joint computation and visualization)
        if os.path.exists(disc_faces_path):
            with open(disc_faces_path, "rb") as f:
                disc_faces = pkl.load(f)
            print(f"Loaded disc faces from {disc_faces_path}")
        elif compute_centroids_if_missing:
            print("Computing disc contact faces for each vertebra...")
            # Use rough centroid-based spine axis for initial disc face detection
            centroids = np.array([m.center_mass for m in meshes])
            disc_faces = find_disc_faces_for_vertebrae_chain(
                meshes, centroids[:-1])
            # Save for future use
            with open(disc_faces_path, "wb") as f:
                pkl.dump(disc_faces, f)
            print(f"Saved disc faces to {disc_faces_path}")
        else:
            disc_faces = [(np.array([]), np.array([]))
                          for _ in vertebrae_names]
            print("No disc faces computed")

        # Compute joint centers and axes
        if use_disc_faces_for_joints and any(len(df[0]) > 0 or len(df[1]) > 0 for df in disc_faces):
            # Use disc face geometry for more accurate joint positioning
            print("Computing joint positions from disc face geometry...")
            mid_points, axes = compute_joints_from_disc_faces(
                meshes, disc_faces)

            # Optionally save the computed values
            if compute_centroids_if_missing:
                np.save(centroids_path, mid_points)
                np.save(axes_path, axes)
                print(f"Saved disc-face-based centroids and axes")
        else:
            # Fallback to centroid-based computation
            if os.path.exists(centroids_path):
                mid_points = np.load(centroids_path)
                print(f"Loaded centroids from {centroids_path}")
            elif compute_centroids_if_missing:
                print("Computing volumetric centroids for joint positions...")
                mid_points = compute_all_joint_centroids(meshes)
                np.save(centroids_path, mid_points)
                print(f"Saved computed centroids to {centroids_path}")
            else:
                raise FileNotFoundError(
                    f"Centroids file not found: {centroids_path}")

            if os.path.exists(axes_path):
                axes = np.load(axes_path)
                print(f"Loaded axes from {axes_path}")
            elif compute_centroids_if_missing:
                print("Computing joint axes from centroids...")
                axes = compute_joint_axes_from_centroids(mid_points)
                np.save(axes_path, axes)
                print(f"Saved computed axes to {axes_path}")
            else:
                raise FileNotFoundError(f"Axes file not found: {axes_path}")

        # Convert disc faces to the per-vertebra pairing format (bottom, top)
        vert_indices_pairing = []
        for tf, bf in disc_faces:
            bottom = bf if len(bf) > 0 else None
            top = tf if len(tf) > 0 else None
            vert_indices_pairing.append((bottom, top))

        # Create vertebra objects
        verts = []
        for ind, name in enumerate(vertebrae_names):
            face_pairing = vert_indices_pairing[ind]
            vert = Vertebra(name, meshes[ind], open3d_meshes[ind],
                            disc_faces_bottom=face_pairing[0], disc_faces_top=face_pairing[1])
            if ind != len(vertebrae_names) - 1:
                vert.initialize_joint(mid_points[ind], axes[ind])

            verts.append(vert)
        self.initialize_links(verts)

    def __init__(self, vertebrae_names, directory_path):
        self.child = None
        self.joint = None
        self.transform = np.eye(4)
        self.fparams = np.zeros(shape=6)
        self.load_and_set_vertebrae(vertebrae_names, directory_path)
        self.get_global_joint()
        self.collision_manager = None
        self.set_collision_manager()

    def initialize_links(self, vertebrae):
        for ind, vertebra in enumerate(vertebrae):
            vertebra.order = ind
            if self.child is None:
                self.child = vertebra
            if ind != 0:
                vertebra.parent = vertebrae[ind - 1]
            if ind != len(vertebrae) - 1:
                vertebra.child = vertebrae[ind + 1]

    def set_collision_manager(self):
        collision_manager = CollisionManagerSpine()
        node = self.child

        while node is not None:
            collision_manager.add_object(node.name, node.mesh)
            node = node.child
        collision_manager.set_initial_collision_depth()
        self.collision_manager = collision_manager

    # this is only called once for the initial rough registration
    def update_transform(self, new_transform):
        self.transform = self.transform @ new_transform

        node = self.child
        while node is not None:
            node.update_transform(new_transform)
            node = node.child

    def get_global_joint(self):
        cent_of_mass_joint = self.child.mesh.center_mass
        global_joint = copy.deepcopy(self.child.joint)
        global_joint.T_joint[:3, 3] = cent_of_mass_joint
        global_joint.parent = self
        self.joint = global_joint

    def propagate_transform(self, new_transform):

        self.child.update_transform(new_transform)
        self.child.propagate_transform(new_transform)

    def compute_candidate_transforms(self, new_transform):
        """
        Compute what the transforms would be after applying new_transform,
        without actually applying them. Used for collision testing.

        Returns:
            dict: mapping of vertebra name to candidate transform matrix
        """
        candidates = {}
        node = self.child
        while node is not None:
            candidate_transform = node.transform @ new_transform
            candidates[node.name] = candidate_transform
            node = node.child
        return candidates

    def apply_candidate_to_collision_manager(self, collision_manager, candidate_transforms):
        """
        Apply candidate transforms to the collision manager for testing.
        """
        for name, transform in candidate_transforms.items():
            collision_manager.set_transform(name, transform)

    def revert_collision_manager(self, collision_manager):
        """
        Revert collision manager transforms to match actual current vertebra transforms.
        """
        node = self.child
        while node is not None:
            collision_manager.set_transform(node.name, node.transform)
            node = node.child

    def sync_collision_transforms(self, collision_manager, new_transform):
        """
        Sync collision manager with actual transforms after a transform is applied.
        """
        node = self.child
        while node is not None:
            collision_manager.set_transform(node.name, node.transform)
            node = node.child

    def get_vertebra_count(self):
        vert = self.child
        count = 0
        while vert is not None:
            count += 1
            vert = vert.child
        return count

    def get_joint_count(self):
        """Return number of joints including global joint (vertebra_count)."""
        # Global joint + (vertebra_count - 1) inter-vertebral joints = vertebra_count
        return self.get_vertebra_count()

    def reset_all_transforms(self):
        """Reset all vertebrae transforms to identity."""
        node = self.child
        while node is not None:
            # Reset o3d mesh to original pose
            node.mesh_o3d.transform(np.linalg.inv(node.transform))
            # Reset transform to identity
            node.transform = np.eye(4)
            # Reset collision manager transform to identity for this vertebra
            self.collision_manager.set_transform(node.name, np.eye(4))
            # Reset joint if it exists
            if node.joint is not None:
                node.joint.center = node.joint.original_center.copy()
                node.joint.axes = [ax.copy()
                                   for ax in node.joint.original_axes]
                node.joint.T_joint = node.joint.original_T_joint.copy()
                node.joint.transform = np.eye(4)
            node = node.child
        # Reset collision manager depth tracking
        self.collision_manager.set_initial_collision_depth()

    def apply_all_fparams(self, fparams, skip_collision_check=False):
        joint_count = self.get_joint_count()
        if len(fparams) != (joint_count * 6):
            raise ValueError(
                f"Fparams len ({len(fparams)}) must be equal to joint_count * 6 ({joint_count * 6})")
        ind = 0

        current_joint = self.joint
        while current_joint is not None and ind < len(fparams):
            # for each joint, get the parent of it
            rel_params = fparams[ind: ind + 6]
            current_joint.propagate_fparam_transform(
                self.collision_manager, rel_params, skip_collision_check)
            # Navigate to the next joint in the chain
            next_vertebra = current_joint.parent.child
            if next_vertebra is not None:
                current_joint = next_vertebra.joint
            else:
                current_joint = None
            ind += 6

    def fetch_o3d_meshes(self):

        node = self.child

        meshes = []
        while node is not None:
            meshes.append(node.mesh_o3d)
            node = node.child

        return meshes

    def fetch_disc_face_meshes(self):
        """
        Create Open3D meshes highlighting the disc contact faces.

        Returns:
            list of o3d.geometry.TriangleMesh: Colored meshes showing disc faces
                - Red: bottom disc faces (inferior endplate)
                - Green: top disc faces (superior endplate)
        """
        import open3d as o3d

        disc_meshes = []
        node = self.child

        while node is not None:
            vertices = np.asarray(node.mesh_o3d.vertices)
            triangles = np.asarray(node.mesh_o3d.triangles)

            # Create mesh for bottom disc faces (red)
            if node.disc_faces_bottom is not None and len(node.disc_faces_bottom) > 0:
                bottom_mesh = o3d.geometry.TriangleMesh()
                # Get the relevant triangles
                bottom_triangles = triangles[node.disc_faces_bottom]
                # Get unique vertex indices
                unique_verts = np.unique(bottom_triangles.flatten())
                # Create vertex index mapping
                vert_map = {old: new for new, old in enumerate(unique_verts)}
                # Remap triangles
                remapped_triangles = np.array(
                    [[vert_map[v] for v in tri] for tri in bottom_triangles])

                bottom_mesh.vertices = o3d.utility.Vector3dVector(
                    vertices[unique_verts])
                bottom_mesh.triangles = o3d.utility.Vector3iVector(
                    remapped_triangles)
                bottom_mesh.paint_uniform_color([1.0, 0.2, 0.2])  # Red
                bottom_mesh.compute_vertex_normals()
                disc_meshes.append(bottom_mesh)

            # Create mesh for top disc faces (green)
            if node.disc_faces_top is not None and len(node.disc_faces_top) > 0:
                top_mesh = o3d.geometry.TriangleMesh()
                top_triangles = triangles[node.disc_faces_top]
                unique_verts = np.unique(top_triangles.flatten())
                vert_map = {old: new for new, old in enumerate(unique_verts)}
                remapped_triangles = np.array(
                    [[vert_map[v] for v in tri] for tri in top_triangles])

                top_mesh.vertices = o3d.utility.Vector3dVector(
                    vertices[unique_verts])
                top_mesh.triangles = o3d.utility.Vector3iVector(
                    remapped_triangles)
                top_mesh.paint_uniform_color([0.2, 1.0, 0.2])  # Green
                top_mesh.compute_vertex_normals()
                disc_meshes.append(top_mesh)

            node = node.child

        return disc_meshes

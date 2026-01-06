"""
Interactive Spine Editor

This script loads vertebrae meshes, creates a Spine object using the 
simplify_kinematic module, and allows interactive manipulation of joint 
angles using keyboard controls.

Controls:
    Number keys (0-9): Select joint to manipulate
    Z/X: Decrease/Increase X rotation
    C/V: Decrease/Increase Y rotation
    B/N: Decrease/Increase Z rotation
    T/Y: Decrease/Increase X translation
    U/I: Decrease/Increase Y translation
    O/P: Decrease/Increase Z translation
    Q: Quit and print final parameters
"""

import open3d as o3d
import numpy as np
import time
from simplify_kinematic import Spine


class InteractiveSpineVisualizer:
    """Interactive visualizer for manipulating spine joint angles."""

    def __init__(self):
        self.vis = o3d.visualization.VisualizerWithKeyCallback()
        self.vis.create_window(window_name="Interactive Spine Editor",
                               width=1280, height=720)

        # Register rotation controls
        self.vis.register_key_callback(ord("Z"), self.decrement_x_rotation)
        self.vis.register_key_callback(ord("X"), self.increment_x_rotation)
        self.vis.register_key_callback(ord("C"), self.decrement_y_rotation)
        self.vis.register_key_callback(ord("V"), self.increment_y_rotation)
        self.vis.register_key_callback(ord("B"), self.decrement_z_rotation)
        self.vis.register_key_callback(ord("N"), self.increment_z_rotation)

        # Register translation controls
        self.vis.register_key_callback(ord("T"), self.decrement_x_translation)
        self.vis.register_key_callback(ord("Y"), self.increment_x_translation)
        self.vis.register_key_callback(ord("U"), self.decrement_y_translation)
        self.vis.register_key_callback(ord("I"), self.increment_y_translation)
        self.vis.register_key_callback(ord("O"), self.decrement_z_translation)
        self.vis.register_key_callback(ord("P"), self.increment_z_translation)

        # Register quit and reset
        self.vis.register_key_callback(ord("Q"), self.quit_callback)
        self.vis.register_key_callback(ord("R"), self.reset_callback)

        # Register disc face toggle
        self.vis.register_key_callback(ord("D"), self.toggle_disc_faces)

        # Register joint visualization toggle
        self.vis.register_key_callback(
            ord("J"), self.toggle_joint_visualization)

        # Register number keys for joint selection
        for i in range(10):
            self.vis.register_key_callback(
                48 + i, self.create_number_callback(i))

        self.spine = None
        self.meshes = []
        self.selected_joint = 0
        self.joint_count = 0
        self.is_running = True

        # Rotation and translation increments (in degrees for rotation, mm for translation)
        self.rotation_increment = 1.0
        self.translation_increment = 1.0

        # Store fparams: 6 values per joint (rx, ry, rz, tx, ty, tz)
        self.fparams = None

        # Coordinate frame for visualization
        self.coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
            size=50.0)

        # Debounce: track last key press time to prevent multiple triggers
        self.last_key_time = 0
        self.debounce_delay = 0.15  # 150ms delay between key presses

        # Disc face visualization
        self.disc_meshes = []
        self.disc_faces_visible = False

        # Joint visualization
        self.joint_geometries = []
        self.joints_visible = False

    def _debounce(self):
        """Check if enough time has passed since last key press. Returns True if action should proceed."""
        current_time = time.time()
        if current_time - self.last_key_time < self.debounce_delay:
            return False
        self.last_key_time = current_time
        return True

    def create_number_callback(self, number):
        """Create a callback for number key presses to select joints."""
        def number_callback(vis):
            if not self._debounce():
                return False
            if number < self.joint_count:
                self.selected_joint = number
                print(f"Selected joint {number}")
            else:
                print(
                    f"Joint {number} does not exist (max: {self.joint_count - 1})")
            return False
        return number_callback

    def get_fparam_index(self):
        """Get the starting index in fparams for the selected joint."""
        return self.selected_joint * 6

    def adjust_param(self, offset, increment):
        """Adjust a parameter and update the visualization incrementally."""
        if not self._debounce():
            return
        idx = self.get_fparam_index() + offset
        if idx < len(self.fparams):
            # Store old value in case we need to revert
            old_value = self.fparams[idx]
            self.fparams[idx] += increment

            # Try to apply the incremental transform with collision checking
            success = self.apply_incremental_transform(offset, increment)

            if success:
                print(
                    f"Joint {self.selected_joint}, param {offset}: {self.fparams[idx]:.2f}")
            else:
                # Revert fparams if collision check failed
                self.fparams[idx] = old_value
                print(
                    f"Joint {self.selected_joint}, param {offset}: BLOCKED by collision (staying at {self.fparams[idx]:.2f})")

    # Rotation callbacks
    def decrement_x_rotation(self, vis):
        self.adjust_param(0, -self.rotation_increment)
        return False

    def increment_x_rotation(self, vis):
        self.adjust_param(0, self.rotation_increment)
        return False

    def decrement_y_rotation(self, vis):
        self.adjust_param(1, -self.rotation_increment)
        return False

    def increment_y_rotation(self, vis):
        self.adjust_param(1, self.rotation_increment)
        return False

    def decrement_z_rotation(self, vis):
        self.adjust_param(2, -self.rotation_increment)
        return False

    def increment_z_rotation(self, vis):
        self.adjust_param(2, self.rotation_increment)
        return False

    # Translation callbacks
    def decrement_x_translation(self, vis):
        self.adjust_param(3, -self.translation_increment)
        return False

    def increment_x_translation(self, vis):
        self.adjust_param(3, self.translation_increment)
        return False

    def decrement_y_translation(self, vis):
        self.adjust_param(4, -self.translation_increment)
        return False

    def increment_y_translation(self, vis):
        self.adjust_param(4, self.translation_increment)
        return False

    def decrement_z_translation(self, vis):
        self.adjust_param(5, -self.translation_increment)
        return False

    def increment_z_translation(self, vis):
        self.adjust_param(5, self.translation_increment)
        return False

    def reset_callback(self, vis):
        """Reset all parameters to zero."""
        self.fparams = np.zeros(self.joint_count * 6)
        print("Reset all parameters to zero")
        self.reapply_all_fparams()
        return False

    def toggle_disc_faces(self, vis):
        """Toggle visibility of disc contact faces."""
        if not self._debounce():
            return False

        if self.disc_faces_visible:
            # Hide disc faces
            for mesh in self.disc_meshes:
                self.vis.remove_geometry(mesh, reset_bounding_box=False)
            self.disc_meshes = []
            self.disc_faces_visible = False
            print("Disc faces hidden")
        else:
            # Show disc faces
            self.disc_meshes = self.spine.fetch_disc_face_meshes()
            for mesh in self.disc_meshes:
                self.vis.add_geometry(mesh, reset_bounding_box=False)
            self.disc_faces_visible = True
            print(f"Disc faces shown ({len(self.disc_meshes)} regions)")

        self.vis.poll_events()
        self.vis.update_renderer()
        return False

    def create_arrow(self, start, direction, length=15.0, color=[1, 0, 0]):
        """Create an arrow mesh from start point in given direction."""
        # Normalize direction
        direction = np.array(direction)
        norm = np.linalg.norm(direction)
        if norm == 0:
            return None
        direction = direction / norm

        # Create cylinder for arrow shaft
        cylinder_radius = length * 0.03
        cone_radius = length * 0.08
        cone_height = length * 0.2
        cylinder_height = length - cone_height

        # Create arrow as cylinder + cone
        cylinder = o3d.geometry.TriangleMesh.create_cylinder(
            radius=cylinder_radius, height=cylinder_height)
        cone = o3d.geometry.TriangleMesh.create_cone(
            radius=cone_radius, height=cone_height)

        # Position cone at end of cylinder
        cone.translate([0, 0, cylinder_height / 2 + cone_height / 2])

        # Combine
        arrow = cylinder + cone

        # Rotate to align with direction (default is along z-axis)
        z_axis = np.array([0, 0, 1])
        if not np.allclose(direction, z_axis) and not np.allclose(direction, -z_axis):
            rotation_axis = np.cross(z_axis, direction)
            rotation_axis = rotation_axis / np.linalg.norm(rotation_axis)
            angle = np.arccos(np.clip(np.dot(z_axis, direction), -1, 1))
            R = o3d.geometry.get_rotation_matrix_from_axis_angle(
                rotation_axis * angle)
            arrow.rotate(R, center=[0, 0, 0])
        elif np.allclose(direction, -z_axis):
            # Flip 180 degrees
            R = o3d.geometry.get_rotation_matrix_from_axis_angle([np.pi, 0, 0])
            arrow.rotate(R, center=[0, 0, 0])

        # Translate to start position
        arrow.translate(start)

        # Color
        arrow.paint_uniform_color(color)
        arrow.compute_vertex_normals()

        return arrow

    def create_joint_geometries(self):
        """Create spheres and axes arrows for all joints at their current transformed positions."""
        geometries = []

        # Get joint info by traversing the spine
        joint_info = []

        # Global joint - this one moves with the whole spine transform
        if self.spine.joint is not None:
            # The global joint is attached to the Spine, apply spine's transform
            original_T_joint = self.spine.joint.original_T_joint
            # Global joint doesn't have a vertebra transform, but the spine itself might have one
            transformed_T_joint = self.spine.transform @ original_T_joint

            joint_info.append({
                'center': transformed_T_joint[:3, 3].copy(),
                'axes': [
                    transformed_T_joint[:3, 0],  # x-axis
                    transformed_T_joint[:3, 1],  # y-axis
                    transformed_T_joint[:3, 2],  # z-axis
                ],
                'name': 'Global'
            })

        # Vertebra joints - each joint moves with its parent vertebra
        node = self.spine.child
        joint_idx = 1
        while node is not None:
            if node.joint is not None:
                # Apply the vertebra's current transform to the joint's original T_joint
                original_T_joint = node.joint.original_T_joint
                transformed_T_joint = node.transform @ original_T_joint

                joint_info.append({
                    'center': transformed_T_joint[:3, 3].copy(),
                    'axes': [
                        transformed_T_joint[:3, 0],
                        transformed_T_joint[:3, 1],
                        transformed_T_joint[:3, 2],
                    ],
                    'name': f'Joint {joint_idx}'
                })
                joint_idx += 1
            node = node.child

        # Create visualization for each joint
        axis_colors = [
            [1, 0, 0],  # X - red
            [0, 1, 0],  # Y - green
            [0, 0, 1],  # Z - blue
        ]

        for idx, info in enumerate(joint_info):
            center = info['center']

            # Create sphere at joint center
            sphere = o3d.geometry.TriangleMesh.create_sphere(radius=3.0)
            sphere.translate(center)
            # Color based on whether it's selected
            if idx == self.selected_joint:
                sphere.paint_uniform_color([1, 1, 0])  # Yellow for selected
            else:
                sphere.paint_uniform_color([0.7, 0.7, 0.7])  # Gray for others
            sphere.compute_vertex_normals()
            geometries.append(sphere)

            # Create arrows for each axis
            for axis_idx, axis in enumerate(info['axes']):
                arrow = self.create_arrow(
                    center, axis, length=12.0, color=axis_colors[axis_idx])
                if arrow is not None:
                    geometries.append(arrow)

        return geometries

    def toggle_joint_visualization(self, vis):
        """Toggle visibility of joint locations and axes."""
        if not self._debounce():
            return False

        if self.joints_visible:
            # Hide joints
            for geom in self.joint_geometries:
                self.vis.remove_geometry(geom, reset_bounding_box=False)
            self.joint_geometries = []
            self.joints_visible = False
            print("Joint visualization hidden")
        else:
            # Show joints
            self.joint_geometries = self.create_joint_geometries()
            for geom in self.joint_geometries:
                self.vis.add_geometry(geom, reset_bounding_box=False)
            self.joints_visible = True
            print(
                f"Joint visualization shown ({len(self.joint_geometries)} geometries)")

        self.vis.poll_events()
        self.vis.update_renderer()
        return False

    def quit_callback(self, vis):
        """Quit and print final parameters."""
        print("\n" + "="*60)
        print("Final fparams:")
        print(self.fparams)
        print("="*60)
        self.is_running = False
        self.vis.destroy_window()
        return False

    def set_spine(self, spine):
        """Set the spine object and initialize visualization."""
        self.spine = spine
        self.meshes = spine.fetch_o3d_meshes()

        # Count joints (vertebrae - 1, since last vertebra has no joint)
        self.joint_count = spine.get_joint_count()

        # Initialize fparams (6 per joint)
        self.fparams = np.zeros(self.joint_count * 6)

        # Add meshes to visualizer
        for mesh in self.meshes:
            mesh.compute_vertex_normals()
            self.vis.add_geometry(mesh)

        # Add coordinate frame
        # self.vis.add_geometry(self.coord_frame)

        # Color the meshes distinctly
        self.color_meshes()

        # Center the view on the meshes
        self.vis.reset_view_point(True)

        print(
            f"\nLoaded spine with {len(self.meshes)} vertebrae and {self.joint_count} joints")
        print("\nControls:")
        print("  0-9: Select joint")
        print("  Z/X: X rotation -/+")
        print("  C/V: Y rotation -/+")
        print("  B/N: Z rotation -/+")
        print("  T/Y: X translation -/+")
        print("  U/I: Y translation -/+")
        print("  O/P: Z translation -/+")
        print("  D: Toggle disc face visualization")
        print("  J: Toggle joint visualization")
        print("  R: Reset all parameters")
        print("  Q: Quit")

    def color_meshes(self):
        """Assign distinct colors to each vertebra mesh."""
        n_meshes = len(self.meshes)
        for i, mesh in enumerate(self.meshes):
            # Generate distinct colors using HSV-like distribution
            hue = i / n_meshes
            color = self.hsv_to_rgb(hue, 0.7, 0.9)
            mesh.paint_uniform_color(color)

    @staticmethod
    def hsv_to_rgb(h, s, v):
        """Convert HSV to RGB color."""
        import colorsys
        r, g, b = colorsys.hsv_to_rgb(h, s, v)
        return [r, g, b]

    def update_visualization(self):
        """Update the visualization after transforms have been applied."""
        # Update meshes in visualizer
        self.meshes = self.spine.fetch_o3d_meshes()
        for mesh in self.meshes:
            self.vis.update_geometry(mesh)

        # Update joint visualization if visible
        if self.joints_visible:
            # Remove old joint geometries
            for geom in self.joint_geometries:
                self.vis.remove_geometry(geom, reset_bounding_box=False)
            # Create new joint geometries at updated positions
            self.joint_geometries = self.create_joint_geometries()
            for geom in self.joint_geometries:
                self.vis.add_geometry(geom, reset_bounding_box=False)

        self.vis.poll_events()
        self.vis.update_renderer()

    def apply_incremental_transform(self, param_offset, increment):
        """
        Apply an incremental transform for a single parameter change.

        Returns:
            bool: True if transform was applied, False if blocked by collision
        """
        # Get the joint for the selected joint index
        joint = self.get_joint_by_index(self.selected_joint)
        if joint is None:
            return False

        # Create fparams array with just this one change
        delta_fparams = np.zeros(6)
        delta_fparams[param_offset] = increment

        # Apply the transform through the joint with collision checking
        # This will test the transform and only apply it if collision check passes
        transform = joint.get_fparam_transform(delta_fparams)

        # Test the transform in collision manager
        candidate_transforms = joint.parent.compute_candidate_transforms(
            transform)
        joint.parent.apply_candidate_to_collision_manager(
            self.spine.collision_manager, candidate_transforms)

        # Debug: print collision info
        current_depth = self.spine.collision_manager.get_collision_depth()
        initial_depth = self.spine.collision_manager.initial_depth
        threshold = initial_depth * 1.05 + 0.1
        print(
            f"  [DEBUG] Penetration depth: {current_depth:.2f}, threshold: {threshold:.2f}, initial: {initial_depth:.2f}")

        if self.spine.collision_manager.is_more_collisions():
            # Collision detected - revert collision manager and don't apply
            joint.parent.revert_collision_manager(self.spine.collision_manager)
            return False

        # Collision check passed - apply the actual transform
        joint.parent.propagate_transform(transform)

        # Update the visualization
        self.update_visualization()
        return True

    def get_joint_by_index(self, joint_index):
        """Get the joint object for a given joint index."""
        if joint_index == 0:
            # Global joint
            return self.spine.joint

        # Navigate to the vertebra joint
        node = self.spine.child
        current_idx = 1
        while node is not None:
            if node.joint is not None:
                if current_idx == joint_index:
                    return node.joint
                current_idx += 1
            node = node.child
        return None

    def reapply_all_fparams(self):
        """Reset and reapply all fparams (used for reset operation)."""
        # Reset spine to original state
        self.spine.reset_all_transforms()

        # Apply all fparams without collision checking (for reset/reload)
        try:
            self.spine.apply_all_fparams(
                self.fparams, skip_collision_check=True)
        except Exception as e:
            print(f"Error applying fparams: {e}")

        self.update_visualization()

    def run(self):
        """Run the interactive visualizer."""
        self.vis.run()


def main():
    # Directory containing vertebrae mesh files
    vertebrae_directory = "/home/connorscomputer/Desktop/01082025_ct_objs"

    # Define vertebrae names based on actual files in the directory
    # These are thoracic vertebrae T7-T10
    vertebrae_names = [
        "01082025_ct_scan_vertebra_14_T7",
        "01082025_ct_scan_vertebra_15_T8",
        "01082025_ct_scan_vertebra_16_T9",
        "01082025_ct_scan_vertebra_17_T10"
    ]

    print(f"Loading vertebrae from: {vertebrae_directory}")
    print(f"Vertebrae to load: {vertebrae_names}")
    print("\nCentroids and axes will be computed automatically if not present.")

    try:
        # Create Spine object
        spine = Spine(vertebrae_names, vertebrae_directory)

        # Create and run interactive visualizer
        visualizer = InteractiveSpineVisualizer()
        visualizer.set_spine(spine)
        visualizer.run()

    except FileNotFoundError as e:
        print(f"\nError: {e}")
        print("\nPlease ensure all required files are present in the directory.")
        print("Required files:")
        print("  - Mesh files for each vertebra")
        print("  - centroids.npy")
        print("  - axes.npy")
        print("  - face_indices.pkl")
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()

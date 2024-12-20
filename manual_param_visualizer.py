import open3d as o3d
import numpy as np


class InteractiveMeshVisualizer:
    def __init__(self):
        self.vis = o3d.visualization.VisualizerWithKeyCallback()
        self.vis.create_window()
        self.vis.register_key_callback(ord("T"), self.decrement_x_transform)
        self.vis.register_key_callback(ord("Y"), self.increment_x_transform)
        self.vis.register_key_callback(ord("U"), self.decrement_y_transform)
        self.vis.register_key_callback(ord("I"), self.increment_y_transform)
        self.vis.register_key_callback(ord("O"), self.decrement_z_transform)
        self.vis.register_key_callback(ord("P"), self.increment_z_transform)

        self.vis.register_key_callback(ord("Z"), self.decrement_x_rotation)
        self.vis.register_key_callback(ord("X"), self.increment_x_rotation)
        self.vis.register_key_callback(ord("C"), self.decrement_y_rotation)
        self.vis.register_key_callback(ord("V"), self.increment_y_rotation)
        self.vis.register_key_callback(ord("B"), self.decrement_z_rotation)
        self.vis.register_key_callback(ord("N"), self.increment_z_rotation)

        self.vis.register_key_callback(ord("Q"), self.quit_callback)
        self.mesh = None
        self.spine_obj = None
        self.last_clicked_node = 0
        self.is_running = True
        self.fparams = None
        self.node_count = 1

    def create_number_callback(self, number):
        def number_callback(vis):
            clicked = max(min(self.node_count - 1, number), 0)
            print(f"Number {clicked} pressed")
            self.last_clicked_node = clicked
            return False
        return number_callback

    def check_link_can_transform(self):
        if self.last_clicked_node is None or self.last_clicked_node != 0:
            return False
        return True

    def update_spine_obj_fparams(self):
        self.spine_obj.reset_spine()
        joint_params = self.fparams[:-6]
        global_params = self.fparams[-6:]
        joint_params, global_params = self.spine_obj.convert_degree_params_to_radians(
            joint_params, global_params)
        self.spine_obj.apply_global_parameters(global_params)
        self.spine_obj.apply_global_transform()
        self.spine_obj.apply_joint_parameters(joint_params)
        self.spine_obj.apply_joint_angles()
        self.mesh.vertices = o3d.utility.Vector3dVector(
            self.spine_obj.vertices)

        self.vis.update_geometry(self.mesh)
        self.vis.update_geometry(self.pcd)
        self.vis.update_renderer()

    def transform_adjust(self, index, increment_value):
        self.fparams[index] += increment_value
        self.update_spine_obj_fparams()

    def get_fparam_trio_index(self):
        if self.last_clicked_node == 0:
            return -6
        return int(self.last_clicked_node - 1) * 3

    def decrement_x_rotation(self, vis):
        self.transform_adjust(self.get_fparam_trio_index(), -0.25)

    def increment_x_rotation(self, vis):
        self.transform_adjust(self.get_fparam_trio_index(), 0.25)

    def decrement_y_rotation(self, vis):
        self.transform_adjust(self.get_fparam_trio_index() + 1, -0.25)

    def increment_y_rotation(self, vis):
        self.transform_adjust(self.get_fparam_trio_index() + 1, 0.25)

    def decrement_z_rotation(self, vis):
        self.transform_adjust(self.get_fparam_trio_index() + 2, -0.25)

    def increment_z_rotation(self, vis):
        self.transform_adjust(self.get_fparam_trio_index() + 2, 0.25)

    def decrement_x_transform(self, vis):
        if self.check_link_can_transform():
            self.transform_adjust(-3, -0.25)

    def increment_x_transform(self, vis):
        if self.check_link_can_transform():
            self.transform_adjust(-3, 0.25)

    def decrement_y_transform(self, vis):
        if self.check_link_can_transform():
            self.transform_adjust(-2, -0.25)

    def increment_y_transform(self, vis):
        if self.check_link_can_transform():
            self.transform_adjust(-2, 0.25)

    def decrement_z_transform(self, vis):
        if self.check_link_can_transform():
            self.transform_adjust(-1, -0.25)

    def increment_z_transform(self, vis):
        if self.check_link_can_transform():
            self.transform_adjust(-1, 0.25)

    def set_mesh_and_spine_obj(self, mesh, spine_obj, fparams, pcd):
        """Load a mesh from file"""
        self.mesh = mesh
        self.pcd = pcd
        self.spine_obj = spine_obj
        self.fparams = fparams
        self.node_count = len(fparams[:-3])//3

        for i in range(10):
            # Use ASCII codes for numbers (48 = '0', 57 = '9')
            self.vis.register_key_callback(
                48 + i, self.create_number_callback(i))
        self.update_spine_obj_fparams()
        self.vis.add_geometry(self.mesh)
        self.vis.add_geometry(self.pcd)

    def quit_callback(self, vis):
        """Quit the application"""
        self.is_running = False
        print(self.fparams)
        self.spine_obj.reset_spine()
        self.vis.destroy_window()

    def run(self):
        """Main run loop"""
        # self.initialize_visualizer()

        # Run the event loop
        self.vis.run()


[3.5,         12.47066159,  11.47931577, -15.45208772,  11.58824518,
 5.75, -10.81504712,  14.42096189,   9.25]

import heapq
import numpy as np
import open3d as o3d
from itertools import permutations


def get_linkage_ordering(centroid_dict):

    total_heap = []
    distance_matrix = np.zeros(shape=(len(centroid_dict), len(centroid_dict)))

    for key, value in centroid_dict.items():
        local_heap = []
        # key is now numeric value

        for other_key, other_value in centroid_dict.items():
            if key != other_key:
                distance = np.linalg.norm(value - other_value)
                local_heap.append(-distance)
                distance_matrix[key][other_key] = distance
        heapq.heapify(local_heap)
        min_val = min(heapq.heappop(local_heap), heapq.heappop(local_heap))
        total_heap.append((min_val, key))

    heapq.heapify(total_heap)
    edge_nodes = [heapq.heappop(total_heap)[1], heapq.heappop(total_heap)[1]]
    node_order = [edge_nodes[0]]

    while len(node_order) < len(centroid_dict):

        current_node = node_order[-1]
        rel_distances = distance_matrix[current_node]
        # this will return the corresponding indexes that give shortest distance to current node
        ind_sorted = np.argsort(rel_distances)
        for i in ind_sorted[1:]:
            if i not in node_order:
                node_order.append(i)
                break
    return node_order


def get_lower_centroid_of_vertebrae_bounding_box(vertices):
    vector_verts = o3d.utility.Vector3dVector(vertices)
    bounding_box = o3d.geometry.OrientedBoundingBox.create_from_points(
        o3d.utility.Vector3dVector(vector_verts))
    candidate_points = None
    i = np.argsort(bounding_box.extent)[-3]

    bbox1, bbox2 = split_oriented_bbox(bounding_box, i)
    p1 = bbox1.get_point_indices_within_bounding_box(vector_verts)
    p2 = bbox2.get_point_indices_within_bounding_box(vector_verts)

    if len(p1) > len(p2):
        return bbox1
        candidate_points = p1
    else:
        return bbox2
        candidate_points = p2


def get_joint_positions(partition_map, mesh):
    # this gets a link
    # I'll modify this, using the bounding box point distribution, only consider half with most points, hopefully gives

    vertices = np.asarray(mesh.vertices)
    bounding_box = get_lower_centroid_of_vertebrae_bounding_box(vertices)
    centroid_dict = {}
    for key, value in partition_map.items():
        rel_vertices = vertices[value]
        rel_rel_vertices = bounding_box.get_point_indices_within_bounding_box(
            o3d.utility.Vector3dVector(rel_vertices))
        centroid_dict[key] = np.mean(rel_vertices[rel_rel_vertices], axis=0)

    link_order = get_linkage_ordering(centroid_dict)
    bone_coords = []
    for ind, i in enumerate(np.array(link_order)):
        if ind > 0 and ind < len(link_order)-1:
            # going from lower vertebrae to current
            prev_vec = (centroid_dict[i] -
                        centroid_dict[link_order[ind-1]])*0.5
            # going from current verterae to next
            next_vec = (centroid_dict[link_order[ind+1]] -
                        centroid_dict[i])*0.5
            # tail of bone goes from center of previous coordinate to halfway out
            tail = centroid_dict[link_order[ind-1]] + prev_vec
            head = centroid_dict[i] + next_vec
            bone_coords.append((tail, head, link_order[ind]))

    l_tail = bone_coords[0][0]
    r_head = bone_coords[-1][1]

    bone_coords.append((r_head, centroid_dict[link_order[-1]], link_order[-1]))
    bone_coords = [(centroid_dict[link_order[0]],
                    l_tail, link_order[0])] + bone_coords

    # each vertebrae has a bone

    return bone_coords


def split_oriented_bbox(bbox: o3d.geometry.OrientedBoundingBox, axis):
    """
    Split an OrientedBoundingBox into two equal parts along its local axis.

    Parameters:
    bbox: open3d.geometry.OrientedBoundingBox to split
    axis: Axis enum indicating which local axis to split along

    Returns:
    tuple: (first_half_bbox, second_half_bbox) - Two new OrientedBoundingBoxes
    """
    # Get the original box parameters
    center = bbox.center
    R = bbox.R  # Rotation matrix
    extent = bbox.extent  # Full dimensions

    # Create new extent for split boxes
    new_extent = extent.copy()
    new_extent[axis] = extent[axis] / 2

    # Calculate offset in the local coordinate system
    offset = np.zeros(3)
    offset[axis] = extent[axis] / 4  # Quarter of the full extent

    # Transform offset to global coordinate system
    global_offset = R @ offset

    # Create the two new boxes
    first_half_center = center - global_offset
    second_half_center = center + global_offset

    first_half_bbox = o3d.geometry.OrientedBoundingBox(
        first_half_center, R, new_extent
    )

    second_half_bbox = o3d.geometry.OrientedBoundingBox(
        second_half_center, R, new_extent
    )

    return first_half_bbox, second_half_bbox

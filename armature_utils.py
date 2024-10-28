import heapq
import numpy as np


def get_linkage_ordering(centroid_dict):

    total_heap = []
    distance_matrix = np.zeros(shape=(len(centroid_dict), len(centroid_dict)))
    ind_tracker = {}
    for ind, (key, value) in enumerate(centroid_dict.items()):
        local_heap = []
        ind_tracker[key] = ind
        for ind2, (other_key, other_value) in enumerate(centroid_dict.items()):
            if key != other_key:
                distance = np.linalg.norm(value - other_value)
                local_heap.append(-distance)
                distance_matrix[ind][ind2] = distance
        heapq.heapify(local_heap)
        min_val = min(heapq.heappop(local_heap), heapq.heappop(local_heap))
        total_heap.append((min_val, key))

    heapq.heapify(total_heap)
    edge_nodes = [heapq.heappop(total_heap)[1], heapq.heappop(total_heap)[1]]
    node_order = [edge_nodes[0]]
    encountered = {edge_nodes[0]}
    reverse_ind_tracker = {v: x for x, v in ind_tracker.items()}

    while len(node_order) < len(centroid_dict):

        current_node = node_order[-1]
        rel_distances = distance_matrix[ind_tracker[current_node]]
        ind_sorted = np.argsort(rel_distances)
        for i in ind_sorted[1:]:
            if reverse_ind_tracker[i] not in encountered:
                node_order.append(reverse_ind_tracker[i])
                encountered.add(reverse_ind_tracker[i])
    return node_order


def get_joint_positions(partition_map, mesh):
    vertices = np.asarray(mesh.vertices)
    centroid_dict = {}
    for key, value in partition_map.items():
        rel_vertices = vertices[value]
        centroid_dict[key] = np.mean(rel_vertices, axis=0)

    link_order = get_linkage_ordering(centroid_dict)
    bone_coords = []
    for ind, i in enumerate(np.array(link_order)):
        if ind > 0 and ind < len(link_order)-1:
            # going from lower vertebrae to current
            prev_vec = (i -
                        link_order[ind-1])*0.5
            # going from current verterae to next
            next_vec = (link_order[ind+1] -
                        i)*0.5
            # tail of bone goes from center of previous coordinate to halfway out
            tail = link_order[ind-1] + prev_vec
            head = i + next_vec
            bone_coords.append((tail, head))

    l_tail = bone_coords[0][0]
    r_head = bone_coords[-1][1]

    bone_coords.append((r_head, link_order[-1]))
    bone_coords = [(link_order[0], l_tail)] + bone_coords

    return bone_coords

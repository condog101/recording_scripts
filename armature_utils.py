import heapq
import numpy as np


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


def get_joint_positions(partition_map, mesh, rgb_num_map):
    # this gets a link
    vertices = np.asarray(mesh.vertices)
    centroid_dict = {}
    for key, value in partition_map.items():
        rel_vertices = vertices[value]
        centroid_dict[rgb_num_map[key]] = np.mean(rel_vertices, axis=0)

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


def get_vector_joint_mappings(tail_heads):
    vectors = []
    for i in range(len(tail_heads), step=2):
        # first tail, and last head is a centroid, no need to half
        # for vectors branching of joint, at head of i[1]
        joint_pos = tail_heads[i][1]
        prev_pos_vec = (tail_heads[i][0] - joint_pos)

        if i != 0:
            prev_pos_vec *= 0.5

        next_post = (tail_heads[i+1][1] - tail_heads[i+1][0])
        if i != len(tail_heads) - 2:
            next_post *= 0.5

        vectors.append((prev_pos_vec, next_post))
    return vectors

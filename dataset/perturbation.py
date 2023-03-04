import csv
import random

from utils import load_data

def perturb_data(method="random", seed=123, perturbation_amount=0):
    """
    Given a networkx graph, returns new version of the graph with edges removed according to the
    given method. Proportion indicates the proportion of edges that are perturbed.
    """

    if perturbation_amount == 0:
        raise Exception("needs to specify perturbation amount")

    random.seed(seed)

    graph, split_dict = load_data()

    match method:
        case "random_remove":
            return random_remove(split_dict, perturbation_amount)
        case "random_add":
            return random_add(split_dict, graph, perturbation_amount)
        case "random_swap":
            return random_swap(split_dict, graph, perturbation_amount)
        case "adversial_remove":
            return adversial_remove(split_dict, graph, perturbation_amount)
        case "adversial_add":
            return adversial_add(split_dict, graph, perturbation_amount)
        case _:
            raise Exception(f"{method} is not a supported method for edge removal.")
        
def adversial_add(split_dict, graph, perturbation_percentages):
    train = split_dict["train"]
    valid = split_dict["valid"]
    test = split_dict["test"]

    edges = train["edge"]
    hidden_edges = test["edge"] # these are all the edges that we want to hide from

    existing_edges = get_existing_edges(train, valid, test)
    all_edges = get_all_possible_edges(graph.number_of_nodes())
    potential_edges = list(all_edges.difference(existing_edges)) # all the edges that we can potentially add

    # first of all, checks whether train or test set spans the entire graph
    print('total number of vertices in graph', graph.number_of_nodes())
    print("train set vertex cover count", get_vertex_cover_cnt(edges)) # train set covers all vertices
    print("test set vertex cover count", get_vertex_cover_cnt(hidden_edges)) # test set only covers a subset of vertices

    # construct adjency list
    adjency_list = {}
    ordered_edge_set = set()
    for edge in edges:
        v1 = edge[0]
        v2 = edge[1]
        if v1 not in adjency_list:
            adjency_list[v1] = set()
        if v2 not in adjency_list:
            adjency_list[v2] = set()
        adjency_list[v1].add(v2)
        adjency_list[v2].add(v1)
        ordered_edge_set.add((min(v1, v2), max(v1, v2)))
    edges = ordered_edge_set

    # get all vertices in test set
    vertices_in_hidden_edges = set()
    ordered_hidden_edges = set() 
    for edge in hidden_edges:
        vertices_in_hidden_edges.add(edge[0])
        vertices_in_hidden_edges.add(edge[1])
        ordered_hidden_edges.add((min(edge[0], edge[1]), max(edge[0], edge[1])))
    hidden_edges = ordered_hidden_edges # make the hidden edge ordered v1 < v2 and into the set form
    
    # filter down list of edges to remove from, we want the edge to touch one vertex from any of the hidden edges
    print("before filtering, there are", len(potential_edges), "edges to consider") 
    filtered_down_edges = set()
    for edge in potential_edges:
        vertex1 = edge[0]
        vertex2 = edge[1]
        if vertex1 in vertices_in_hidden_edges or vertex2 in vertices_in_hidden_edges:
            filtered_down_edges.add((min(vertex1, vertex2), max(vertex1, vertex2)))
    print("after filtering, there are", len(filtered_down_edges), "edges to consider")    

    # intialize edge scores
    edge_scores = {}
    for edge in filtered_down_edges:
        vertex1 = min(edge[0], edge[1])
        vertex2 = max(edge[0], edge[1])
        edge_scores[(vertex1, vertex2)] = 0
    
    vertices_set = set(range(0, graph.number_of_nodes()))
    
    idx = 0
    for edge in filtered_down_edges:
        if idx == 10000: 
            break
        print('iteration', idx)
        vertex1 = edge[0]
        vertex2 = edge[1]
        negative_edge_score = False
        for vertex3 in vertices_set:
            # for edge XY and any vertex in the graph V, for edges XV and YV
            # if one of the edges is existing edge and the other is a hidden edge
            # XV, then it will make the hidden edge more exposed, thus we assign a negative
            # score to it to indicate it is a bad candidate.
            edge1 = (min(vertex1, vertex3), max(vertex1, vertex3))
            edge2 = (min(vertex2, vertex3), max(vertex2, vertex3))
            if edge1 in hidden_edges and edge2 in edges:
                negative_edge_score = True
                break
            if edge1 in edges and edge2 in hidden_edges:
                negative_edge_score = True
                break
        if negative_edge_score:
            edge_scores[edge] = -1
        else:
            # otherwise, the score is assigned the union of X and Y's neighbors (the more neighbors, the more
            # likely other links will be chosen over the hidden link)
            vertex1_neighbors = adjency_list[vertex1]
            vertex2_neighbors = adjency_list[vertex2]
            edge_scores[edge] = len(vertex1_neighbors.union(vertex2_neighbors))
        idx += 1

    # sort edges based on descending edge scores
    edge_scores = list(edge_scores.items())
    edge_scores.sort(key=lambda e : -e[1])

    non_negative_one_cnt = 0
    for edge_score in edge_scores:
        if edge_score[1] != -1:
            non_negative_one_cnt = non_negative_one_cnt + 1
    print("there are", non_negative_one_cnt, "non-negative scores out of", len(edge_scores), "total edges")
    print('top five scores', edge_scores[0:5])

    added_edges = [e[0] for e in edge_scores]

    # loop over each percentage
    for perturbation_percentage in perturbation_percentages:

        # take the [k:] remaining edges, where k is derived from the preturbation percentage
        added_cnt = int(perturbation_percentage * len(edges))
        selected_edges = added_edges[0:added_cnt]

        # write to output file
        file_name = f"dataset/perturbation/adversial_add_{perturbation_percentage}.csv"
        with open(file_name, 'w', newline='') as file:
            writer = csv.writer(file)
            for edge in selected_edges:
                writer.writerow(edge)
    
def adversial_remove(split_dict, graph, perturbation_percentage):
    train = split_dict["train"]
    test = split_dict["test"]

    edges = train["edge"] # these are all the edges that we want to remove from 
    hidden_edges = test["edge"] # these are all the edges that we want to hide from
    
    # intialize edge scores
    edge_scores = {}
    for edge in edges:
        vertex1 = min(edge[0], edge[1])
        vertex2 = max(edge[0], edge[1])
        edge_scores[(vertex1, vertex2)] = 0

    # first of all, checks whether train or test set spans the entire graph
    print('total number of vertices in graph', graph.number_of_nodes())
    print("train set vertex cover count", get_vertex_cover_cnt(edges)) # train set covers all vertices
    print("test set vertex cover count", get_vertex_cover_cnt(hidden_edges)) # test set only covers a subset of vertices

    # construct adjency list
    adjency_list = {}
    for edge in edges:
        v1 = edge[0]
        v2 = edge[1]
        if v1 not in adjency_list:
            adjency_list[v1] = set()
        if v2 not in adjency_list:
            adjency_list[v2] = set()
        adjency_list[v1].add(v2)
        adjency_list[v2].add(v1)

    # get all vertices in test set
    vertices_in_hidden_edges = set()
    for edge in hidden_edges:
        vertices_in_hidden_edges.add(edge[0])
        vertices_in_hidden_edges.add(edge[1])
    
    # filter down list of edges to remove from, we want the edge to touch one vertex from any of the hidden edges
    filtered_down_edges = set()
    for edge in edges:
        vertex1 = edge[0]
        vertex2 = edge[1]
        if vertex1 in vertices_in_hidden_edges or vertex2 in vertices_in_hidden_edges:
            filtered_down_edges.add((min(vertex1, vertex2), max(vertex1, vertex2)))

    # for each hidden edge, examine the common neihbors and check if a triangle is formed
    for edge in hidden_edges:
        vertex1 = edge[0]
        vertex2 = edge[1]
        common_neighbors = adjency_list[vertex1].intersection(adjency_list[vertex2])
        for n in common_neighbors:
            edge1 = (min(vertex1, n), max(vertex1, n))
            edge2 = (min(vertex2, n), max(vertex2, n))
            if edge1 in filtered_down_edges and edge2 in filtered_down_edges:
                edge_scores[edge1] += 1
                edge_scores[edge2] += 1

    # sort edges based on descending edge scores
    edge_scores = list(edge_scores.items())
    edge_scores.sort(key=lambda e : -e[1])

    # take the [k:] remaining edges, where k is derived from the preturbation percentage
    removed_cnt = int(perturbation_percentage * len(edges))
    remaining_edges = edge_scores[removed_cnt:]
    remaining_edges = [e[0] for e in remaining_edges]

    # write to output file
    file_name = f"dataset/perturbation/adversial_remove_{perturbation_percentage}.csv"
    with open(file_name, 'w', newline='') as file:
        writer = csv.writer(file)
        for new_edge in remaining_edges:
            writer.writerow(new_edge)

def get_vertex_cover_cnt(edges: list):
    vertex_set = set()
    for edge in edges:
        vertex_set.add(edge[0])
        vertex_set.add(edge[1])
    return len(vertex_set)

def random_remove(split_dict, perturbation_percentage):
    """
    Randomly removing k edges involves selecting k edges uniformly at random
    and removing those from the training data.
    """

    train_edges = split_dict["train"]["edge"]
    old_edges_cnt = len(train_edges)

    removed_cnt = int(perturbation_percentage * old_edges_cnt)
    random.shuffle(train_edges) 
    new_edges = train_edges[removed_cnt:]

    file_name = f"dataset/perturbation/random_remove_{perturbation_percentage}.csv"
    with open(file_name, 'w', newline='') as file:
        writer = csv.writer(file)
        for new_edge in new_edges:
            writer.writerow(new_edge)

def random_add(split_dict, graph, perturbation_percentage):
    """
    Randomly adding k edges involves choosing k edges that are not in the list
    of edges used for training uniformly at random and adding those to the training data. 
    """

    train = split_dict["train"]
    valid = split_dict["valid"]
    test = split_dict["test"]

    existing_edges = get_existing_edges(train, valid, test)
    all_edges = get_all_possible_edges(graph.number_of_nodes())
    potential_edges = list(all_edges.difference(existing_edges))

    random.shuffle(potential_edges) 
    new_edges_cnt = int(len(train["edge"]) * perturbation_percentage)
    new_edges = potential_edges[0: new_edges_cnt]

    # TODO: make sure you also append the original set of edges at the end of the generated csv
    file_name = f"dataset/perturbation/random_add_{perturbation_percentage}.csv"
    with open(file_name, 'w', newline='') as file:
        writer = csv.writer(file)
        for new_edge in new_edges:
            writer.writerow(new_edge)

def random_swap(split_dict, graph, num_edges_to_swap):
    """
    Randomly swapping k edges involves selecting k pairs of edges (u1, v1) and (u2, v2)
    uniformly at random, removing these from the data, and adding (u2, v1) and (u1, v2), 
    effectively swapping their endpoints 
    """
    
    train = split_dict["train"]
    valid = split_dict["valid"]
    test = split_dict["test"]
    train_edges = train["edge"]

    existing_edges = get_existing_edges(train, valid, test)
    all_edges = get_all_possible_edges(graph.number_of_nodes())
    potential_edges = list(all_edges.difference(existing_edges))

    count = 0 # how many times we have seapped so far
    deleted = set() # all the edges that need to be removed at the end
    added = set() # all the edges that need to be addeda at the end

    visited = set()


    while True:
        if count == num_edges_to_swap: 
            break
        else:
            
            # pick two random edges and see if they are eligible for swapping
            first_edge_idx = random.randint(0, len(train_edges) - 1)
            second_edge_idx = random.randint(0, len(train_edges) - 1)

            # ignore if already visited before
            smaller_idx = min(first_edge_idx, second_edge_idx)
            larger_idx = max(first_edge_idx, second_edge_idx)
            if (smaller_idx, larger_idx) in visited:
                continue

            # make sure starting vertex < ending vertex for first and second edge
            first_edge_i = min(train_edges[first_edge_idx][0], train_edges[first_edge_idx][1])
            first_edge_j = max(train_edges[first_edge_idx][0], train_edges[first_edge_idx][1])
            second_edge_i = min(train_edges[second_edge_idx][0], train_edges[second_edge_idx][1])
            second_edge_j = max(train_edges[second_edge_idx][0], train_edges[second_edge_idx][1])

            # do not consider if any of the vertices are equal to each other
            if first_edge_i == second_edge_i or first_edge_j == second_edge_j \
            or first_edge_i == second_edge_j or first_edge_j == second_edge_i:
                continue

            # generate two new edges by swapping
            first_new_edge = (min(first_edge_i, second_edge_j), max(first_edge_i, second_edge_j))
            second_new_edge = (min(first_edge_j, second_edge_i), max(first_edge_j, second_edge_i))

            # update count, update added and deleted set
            if first_new_edge in potential_edges and second_new_edge in potential_edges:
                deleted.add((train_edges[first_edge_idx][0], train_edges[first_edge_idx][1]))
                deleted.add((train_edges[second_edge_idx][0], train_edges[second_edge_idx][1]))
                added.add(first_new_edge)
                added.add(second_new_edge)
                count = count + 1
                visited.add((smaller_idx, larger_idx))

                print('Found, current count', count)
        
    new_existing_edges = set()
    for edge in train_edges:
        new_existing_edges.add((edge[0], edge[1]))

    # remove all edges from deleted set
    for d in deleted:
        new_existing_edges.remove(d)
    
    # add all edges from added set
    for a in added:
        new_existing_edges.add(a)

    file_name = f"dataset/perturbation/random_swap_{num_edges_to_swap}.csv"
    with open(file_name, 'w', newline='') as file:
        writer = csv.writer(file)
        for new_edge in train_edges:
            writer.writerow(new_edge)
    

# Get and return all possible edges based on the number of the nodes in the graph 
# with constraint (start < end) for all vertices
def get_all_possible_edges(num_of_nodes):
    all_edges = set()
    for i in range(num_of_nodes):
        for j in range(i + 1, num_of_nodes):
            all_edges.add((i, j))
    return all_edges


# Get existing edges that are considered by the train, validation, and test set
# This also includes the negative edges with constraint (start < end) for all vertices
def get_existing_edges(train, valid, test):
    existing_edges = train["edge"] + valid["edge"] + valid["edge_neg"] + test["edge"] + test["edge_neg"]
    new_existing_edges = set()
    for existing_edge in existing_edges:
        smaller_node = min(existing_edge[0], existing_edge[1])
        bigger_node = max(existing_edge[0], existing_edge[1])
        new_existing_edges.add((smaller_node, bigger_node))
    return new_existing_edges

if __name__ == "__main__":
    
    # option: random_add(% of edges added), random_remove(% of edges removed), random_swap(num_of_edges_to_swap)
    # adversial_remove(% of edges removed)
    perturbation_amount = [0.01, 0.1, 0.25, 0.5, 1.0]
    perturb_data(method="adversial_add", perturbation_amount=perturbation_amount)


    # Comment this out to know how many edges exist in train, test, and valid, along with some other info
    # graph, split_dict = load_data()
    # train = split_dict["train"]
    # valid = split_dict["valid"]
    # test = split_dict["test"]
    # pos_edge_cnt = len(train["edge"]) + len(test["edge"]) + len(valid["edge"])
    # neg_egde_cnt = len(test["edge_neg"]) + len(valid["edge_neg"])
    # total_number_edges = graph.number_of_edges()
    # total_number_nodes = graph.number_of_nodes()
    # print('positive edges', len(train["edge"]), len(test["edge"]),len(valid["edge"] ))
    # print('negative edges', neg_egde_cnt)
    # print('total number of edges', total_number_edges)
    # print('total number of nodes', total_number_nodes)
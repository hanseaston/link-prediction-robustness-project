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

    # print(len(split_dict["train"]["edge"]))

    match method:
        case "random_remove":
            return random_remove(split_dict, perturbation_amount)
        case "random_add":
            return random_add(split_dict, graph, perturbation_amount)
        case "random_swap":
            return random_swap(split_dict, graph, perturbation_amount)
        case _:
            raise Exception(f"{method} is not a supported method for edge removal.")

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
    
    perturbation_amount = 10000
    # option: random_add(% of edges added), random_remove(% of edges removed), random_swap(num_of_edges_to_swap)
    perturb_data(method="random_swap", perturbation_amount=perturbation_amount)


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
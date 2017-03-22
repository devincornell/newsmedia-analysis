
from os import walk
import itertools
import functools

def get_network_filenames(folder='results', ext='.gexf', ex_ext='_sparse.gexf'):
    # get model filenames
    files = dict()
    for (dirpath, dirnames, filenames) in walk(folder):
        for file in filenames:
            if file[-len(ext):] == ext and file[-len(ex_ext):] != ex_ext:
                files.append(folder + file)
    return files


def get_sparse_network_files(folder='results/', ext='_sparse.gexf'):
    files = list()
    for (dirpath, dirnames, filenames) in walk(folder):
        for file in filenames:
            if file[-len(ext):] == ext:
                files.append(folder + file)



if __name__ == "__main__":

    graphs_folder = 'results/'
    graph_ext = '.gexf'
    exclude_ext = '_sparse.gexf'
    num_nodes_retained = 20

    g_fnames = get_network_filenames()
    sg_fnames = get_sparse_network_files()


    ## Look at sparse graphs to get top nodes
    nodesets = set()
    for fname in sg_fnames:
        print('Loading network', fname)
        G = nx.read_gexf(fname)

        print('Now removing nodes that are least central: keeping {}', num_nodes_retained)
        G = sn.drop_nodes(G,'eigcent', number=num_nodes_retained, keep_largest=True, verbose=True)
        nodesets.add(set(G.nodes()))

    # reduce using set union
    keepnodes = functools.reduce(lambda x,y: x | y, nodesets)
    print('Keeping {} nodes that are top {} in at least one source.'.format(len(keepnodes), num_nodes_retained))



    # look at non-sparse graphs
    graphs = list()
    for fname in sg_fnames:
        print('Loading network', fname)
        G = nx.read_gexf(fname)
        G.remove_nodes_from(set(G.nodes()) - keepnodes)




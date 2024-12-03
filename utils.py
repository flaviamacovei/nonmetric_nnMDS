import matplotlib.pyplot as plt

def visualise_embeddings(data, model_results, alg_results):
    colours = ['red', 'blue', 'green', 'orange', 'black', 'yellow']

    data = data.to('cpu')
    model_results = model_results.to('cpu')
    alg_results = alg_results.to('cpu')

    fig, ax = plt.subplots()

    xs = model_results[:, 0].float()
    ys = model_results[:, 1].float()
    ax.scatter(xs, ys)#, c = colours[:len(xs)])
    ax.set_aspect('equal')
    plt.savefig("out/2d_model.png")
    plt.clf()

    fig, ax = plt.subplots()

    xs = alg_results[:, 0].float()
    ys = alg_results[:, 1].float()
    ax.scatter(xs, ys)#, c = colours[:len(xs)])
    ax.set_aspect('equal')
    plt.savefig("out/2d_smacof.png")
    plt.clf()

    fig = plt.figure()
    ax = fig.add_subplot(projection = '3d')
    xs = data[:, 0]
    ys = data[:, 1]
    zs = data[:, 2]
    ax.scatter(xs, ys, zs)#, c = colours[:len(xs)])
    ax.set_aspect('equal')
    plt.savefig("out/3d_points.png")
    plt.close('all')

def visualise(results, name):
    colours = ['red', 'blue', 'green', 'orange', 'black', 'yellow']
    results = results.to('cpu')
    fig, ax = plt.subplots()

    xs = results[:, 0].float()
    ys = results[:, 1].float()
    ax.scatter(xs, ys)#, c = colours[:len(xs)])
    ax.set_aspect('equal')
    plt.savefig(f"out/{name}.png")
    plt.clf()
    plt.close('all')

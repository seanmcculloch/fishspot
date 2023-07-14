import numpy as np
from skimage.feature import blob_log
from ClusterWrap.decorator import cluster
from itertools import product
import time
import traceback



# TODO: potential major improvement - after finding coordinates
#       with LoG filter, template match the PSF to the region around
#       each detected point (Fourier Phase Correlation maybe?).
#       upsample the PSF and data to achieve subvoxel accuracy.


def detect_spots_log(
    image,
    min_radius,
    max_radius,
    num_sigma=5,
    **kwargs,
):
    """
    """

    # ensure iterable radii
    if not isinstance(min_radius, (tuple, list, np.ndarray)):
        min_radius = (min_radius,)*image.ndim
    if not isinstance(max_radius, (tuple, list, np.ndarray)):
        max_radius = (max_radius,)*image.ndim

    # compute defaults
    min_radius = np.array(min_radius)
    max_radius = np.array(max_radius)
    min_sigma = 0.8 * min_radius / np.sqrt(image.ndim)
    max_sigma = 1.2 * max_radius / np.sqrt(image.ndim)

    # set given arguments
    kwargs['min_sigma'] = min_sigma
    kwargs['max_sigma'] = max_sigma
    kwargs['num_sigma'] = num_sigma

    # set additional defaults
    if 'overlap' not in kwargs:
        kwargs['overlap'] = 1.0
    if 'threshold' not in kwargs:
        kwargs['threshold'] = None
        kwargs['threshold_rel'] = 0.1

    # run
    return blob_log(image, **kwargs)

@cluster
def distributed_detect_spots_log(
    image,
    min_radius,
    max_radius,
    num_sigma=5,
    blocksize=[512]*3,
    cluster=None,
    cluster_kwargs={},
    **kwargs,
):
    """ 
    """
    print("IN DISTRIBUTED SPOTS LOG", flush=True)

    # ensure iterable radii
    if not isinstance(min_radius, (tuple, list, np.ndarray)):
        min_radius = (min_radius,)*image.ndim
    if not isinstance(max_radius, (tuple, list, np.ndarray)):
        max_radius = (max_radius,)*image.ndim

    # compute defaults
    min_radius = np.array(min_radius)
    max_radius = np.array(max_radius)
    min_sigma = 0.8 * min_radius / np.sqrt(image.ndim)
    max_sigma = 1.2 * max_radius / np.sqrt(image.ndim)

    # set given arguments
    kwargs['min_sigma'] = min_sigma
    kwargs['max_sigma'] = max_sigma
    kwargs['num_sigma'] = num_sigma

    # set additional defaults
    if 'overlap' not in kwargs:
        overlap = int(2*np.max(max_radius))
        kwargs['overlap'] = overlap
    if 'exclude_border' not in kwargs:
        kwargs['exclude_border'] = overlap

    if 'threshold' not in kwargs:
        kwargs['threshold'] = None
        kwargs['threshold_rel'] = 0.1

    # compute number of blocks
    nblocks = np.ceil(np.array(image.shape) / blocksize).astype(int)

 # determine indices for blocking
    print("getting blocks", flush=True)
    indices = []
    for (i, j, k) in product(*[range(x) for x in nblocks]):
        start = np.array(blocksize) * (i, j, k) - overlap
        stop = start + blocksize + 2 * overlap
        start = np.maximum(0, start)
        stop = np.minimum(image.shape, stop)
        coords = tuple(slice(x, y, 1) for x, y in zip(start, stop))

        # add coordinate slices to list
        indices.append(coords)
    print("Defining Block Function", flush=True)
    print("indicies", indices, flush=True)
    def block_blob_log(block, coords, **kwargs):
        print("BLOCKED BLOB LOG", coords, flush=True)
        print(kwargs, flush=True)
        try:
            spots = blob_log(block, **kwargs)
                # adjust for block origin
            origin = np.array([x.start for x in coords])
            spots[:, :3] = spots[:, :3] + origin
        except:
            print("ERROR", flush=True)
                # printing stack trace
            traceback.print_exc()
            spots = np.zeros((1, 6))

        print("Spots Shape:", spots.shape, flush=True)

        return spots
       

    #     # wait for at least one worker to be fully instantiated
    # while ((cluster.client.status == "running") and
    #        (len(cluster.client.scheduler_info()["workers"]) < 1)):
    #     time.sleep(1.0)
    print("SENDING BLOCK BLOB LOG", flush=True)
    print(len(indices), flush=True)
    # for index in indices:
    #     print("BLOCK", index)
    # spots = cluster.client.gather(
    #     cluster.client.map(block_blob_log, indices, batch_size=1, **kwargs)
    # )
    # spots = []
    # for i in range(len(indices)):
    #     print("starting block ", i, "/", len(indices), indices[i])
    #     spots.extend(block_blob_log(image[indices[i]], indices[i], **kwargs))

    import os
    os.environ['DASK_DISTRIBUTED__WORKER__RESOURCES__Foo']="1"
    os.environ['MALLOC_TRIM_THRESHOLD_']="0"
    # spots_all = []
    # for i in range(len(indices)):
    #     spots = cluster.client.gather(
    #         cluster.client.map(block_blob_log, [image[indices[i]]], [indices[i]], batch_size=1, **kwargs, resources={'Foo': 1})
    #     )
    #     spots_all.extend(spots)
    # print("got spots")
    # spots = np.vstack(spots_all)
    # print("returning!")

    spots_all = []
    for i in range(0, len(indices), 8):
        j = i+8
        j = min(j, len(indices))
        spots = cluster.client.gather(
            cluster.client.map(block_blob_log, [image[coords] for coords in indices[i:j]], [coords for coords in indices[i:j]], batch_size=4, **kwargs, resources={'Foo': 1}),
            errors='skip'
        )
        spots_all.extend(spots)
    print("got spots")
    spots = np.vstack(spots_all)
    print("returning!")


    # spots_all = []
    # for i in range(len(indices)):
    #     spots = cluster.client.gather(
    #         cluster.client.submit(block_blob_log, image[indices[i]], indices[i], **kwargs)
    #     )
    #     spots_all.extend(spots)
    # print("got spots")
    # spots = np.vstack(spots_all)
    # print("returning!")


    # spots = cluster.client.gather(
    #     cluster.client.map(block_blob_log, [image[coords] for coords in indices], [coords for coords in indices], batch_size=1, **kwargs, resources={'Foo': 1})
    # )
    # print("got spots")
    # spots = np.vstack(spots)
    # print("returning!")
    return spots
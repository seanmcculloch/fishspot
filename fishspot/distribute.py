import numpy as np
import dask.delayed as delayed
from ClusterWrap.decorator import cluster
import time
from itertools import product
import fishspot.filter as fs_filter
import fishspot.psf as fs_psf
import fishspot.detect as fs_detect


def normalize_data_for_plotting(zarr_data):
    import copy
    z_normalized = copy.deepcopy(zarr_data.astype(np.float32))


    z_normalized[z_normalized<90] = 90

    z_normalized = z_normalized - np.percentile(z_normalized, 5)
    
    z_normalized = z_normalized / np.percentile(z_normalized, 95)
    
    z_normalized[z_normalized<0] = 0
    
    z_normalized[z_normalized>1] = 1

    #swap x and y for plotting
    z_normalized = np.moveaxis(z_normalized, [0,1,2], [1,0,2])

    return z_normalized

def plot_volume_gif(fix_zarr_data, file_path, name, coords=None):
    from matplotlib import pyplot as plt
    import os

    temp_image_folder = os.path.join(file_path,name,'frames')
    
    if not os.path.exists(temp_image_folder):
        os.makedirs(temp_image_folder)
    
    # normalize for display
    f_normalized = normalize_data_for_plotting(fix_zarr_data)

    frame_num = 0
    frame_filenames = []
    frame_copies = 1
    step=1
    for i in range(0, f_normalized.shape[2] , step):
        

        # make RGB version
        a_rgb = np.zeros(f_normalized.shape[:-1] + (3,))
        a_rgb[..., 0] = f_normalized[..., i] 
        a_rgb[..., 1] = f_normalized[..., i] 
        a_rgb[..., 2] = f_normalized[..., i] 


        # create figure
            
        frame_ratio = a_rgb.shape[1] / a_rgb.shape[0]

        for j in range(0, frame_copies):
            if frame_copies > 1:
                frame_filename = temp_image_folder+"/{}_{}.png".format(i, j)
            else:
                frame_filename = temp_image_folder+"/{}.png".format(i)

            

            fig = plt.figure(figsize=(8, int(8*frame_ratio)))
            plt.xlabel("X")
            plt.ylabel("Y")
            if coords is not None:
                plt.imshow(a_rgb, extent=[coords[2].start, coords[2].stop, coords[1].start, coords[1].stop])
            plt.savefig(frame_filename)

            plt.close(fig)
            frame_num += 1
            frame_filenames.append(frame_filename)


    # build gif
    import imageio
    with imageio.get_writer(file_path+name+'.gif', mode='I') as writer:
        for filename in frame_filenames:
            image = imageio.v2.imread(filename)
            writer.append_data(image)
            # Remove files
    # if not config['keep_gif_frames']:
    #     for filename in set(frame_filenames):
    #         os.remove(filename)
    #     os.removedirs(temp_image_folder)



@cluster
def distributed_spot_detection(
    array, blocksize,
    white_tophat_args={},
    psf_estimation_args={},
    deconvolution_args={},
    spot_detection_args={},
    intensity_threshold=0,
    mask=None,
    psf=None,
    psf_retries=3,
    cluster=None,
    cluster_kwargs={},
):
    """
    """
    print("IN DISTRIBUTED BLOB")
    # set white_tophat defaults
    if 'radius' not in white_tophat_args:
        white_tophat_args['radius'] = 4

    # set psf estimation defaults
    if 'radius' not in psf_estimation_args:
        psf_estimation_args['radius'] = 9

    # set spot detection defaults
    if 'min_radius' not in spot_detection_args:
        spot_detection_args['min_radius'] = 1
    if 'max_radius' not in spot_detection_args:
        spot_detection_args['max_radius'] = 6

    # compute overlap depth
    all_radii = [white_tophat_args['radius'],
                 psf_estimation_args['radius'],
                 spot_detection_args['max_radius'],]
    overlap = int(2*max(np.max(x) for x in all_radii))

    # don't detect spots in the overlap region
    if 'exclude_border' not in spot_detection_args:
        spot_detection_args['exclude_border'] = overlap

    # compute mask to array ratio
    if mask is not None:
        ratio = np.array(mask.shape) / array.shape
        stride = np.round(blocksize * ratio).astype(int)

    # compute number of blocks
    nblocks = np.ceil(np.array(array.shape) / blocksize).astype(int)

    # determine indices for blocking
    indices, psfs = [], []
    for (i, j, k) in product(*[range(x) for x in nblocks]):
        start = np.array(blocksize) * (i, j, k) - overlap
        stop = start + blocksize + 2 * overlap
        start = np.maximum(0, start)
        stop = np.minimum(array.shape, stop)
        coords = tuple(slice(x, y) for x, y in zip(start, stop))

        # determine if block is in foreground
        if mask is not None:
            mo = stride * (i, j, k)
            mask_slice = tuple(slice(x, x+y) for x, y in zip(mo, stride))
            if not np.any(mask[mask_slice]): continue

        # add coordinate slices to list
        indices.append(coords)
        psfs.append(psf)

    # pipeline to run on each block
    def detect_spots_pipeline(coords, psf):
        import uuid 
        uuid.uuid4().hex[:6].upper()
        import os

        # load data, background subtract, deconvolve, detect blobs
        block = array[coords]
        wth = fs_filter.white_tophat(block, **white_tophat_args)
        if psf is None:
            # automated psf estimation with error handling
            for i in range(psf_retries):
                try:
                    print("TRYING PSF ESTIMATION")
                    psf = fs_psf.estimate_psf(wth, **psf_estimation_args)
                    print("GOT PSF")
                    print(psf)
                except ValueError:
                    if psf is None:
                        uuid_str = uuid.uuid4().hex[:6].upper()
                        print(np.max(block), np.min(block), np.mean(block), block.shape, coords, os.getpid(),uuid_str, flush=True)
                        plot_volume_gif(block, os.environ['jobdir'], f'psf_block_{os.getpid()}'+uuid_str, coords)
                        if 'inlier_threshold' not in psf_estimation_args:
                            psf_estimation_args['inlier_threshold'] = 0.9
                        psf_estimation_args['inlier_threshold'] -= 0.1
                        print("ERROR WITH PSF ESTIMATION", flush=True)
                    else: break
        decon = fs_filter.rl_decon(wth, psf, **deconvolution_args)
        spots = fs_detect.detect_spots_log(decon, **spot_detection_args)

        # if no spots are found, ensure consistent format
        if spots.shape[0] == 0:
            return np.zeros((0, 7)), psf
        else:
            # append image intensities
            spot_coords = spots[:, :3].astype(int)
            intensities = block[spot_coords[:, 0], spot_coords[:, 1], spot_coords[:, 2]]
            spots = np.concatenate((spots, intensities[..., None]), axis=1)
            spots = spots[ spots[..., -1] > intensity_threshold ]

            # adjust for block origin
            origin = np.array([x.start for x in coords])
            spots[:, :3] = spots[:, :3] + origin
            return spots, psf
    # END: CLOSURE

    # wait for at least one worker to be fully instantiated
    while ((cluster.client.status == "running") and
           (len(cluster.client.scheduler_info()["workers"]) < 1)):
        time.sleep(1.0)

    # submit all alignments to cluster
    spots_and_psfs = cluster.client.gather(
        cluster.client.map(detect_spots_pipeline, indices, psfs)
    )

    # reformat to single array of spots and single psf
    spots, psfs = [], []
    for x, y in spots_and_psfs:
        spots.append(x)
        psfs.append(y)
    spots = np.vstack(spots)
    psf = np.mean(psfs, axis=0)

    # filter with foreground mask
    if mask is not None:
        spots = fs_filter.apply_foreground_mask(
            spots, mask, ratio,
        )

    # return results
    return spots, psf

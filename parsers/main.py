import numpy as np

import geom
import point_cloud


def step2():
    # Read the point cloud into memory.
    pt_cloud = point_cloud.PointCloud("data/tests/test_src.las")
    # Crop the point cloud to a 500x500 meter bounding box centered at it.
    pt_cloud.crop(
        geom.BoundingBox(
            geom.Point(
                0.5 * (np.min(pt_cloud.data.x) + np.max(pt_cloud.data.x)) - 250,
                0.5 * (np.min(pt_cloud.data.y) + np.max(pt_cloud.data.y)) - 250,
            ),
            500,
        )
    )
    # Save the resulting dataset.
    pt_cloud.save("data/tests/test_crp.las")


def step3():
    # Read the point cloud into memory.
    pt_cloud = point_cloud.PointCloud("data/tests/test_crp.las")
    # Classify the ground points of the point cloud.
    pt_cloud.csf("data/tests/csf/test_csf.npy")
    # Save the resulting dataset.
    pt_cloud.save("data/tests/csf/test_csf.las")


def step4():
    # Read the point cloud into memory.
    pt_cloud = point_cloud.PointCloud("data/crp.las")
    pt_cloud.downsample(0.1, method=point_cloud.DownsamplingMethod.RANDOM)
    # Create the digital terrain model of the point cloud and save the resulting dataset.
    pt_cloud.dtm(
        "dtm_0_5_cls_0_9_rnd.tif",
        resolution=0.5,
        method=point_cloud.DTMGenerationMethod.CLASSIFICATION,
    )


if __name__ == "__main__":
    # step2()
    # step3()
    step4()

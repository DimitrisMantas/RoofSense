# Introduction

Our program generally works in a destructive manner, similar to NumPy. This means that each operation performed directly
on a certain point cloud *permanently* alters its state. For example, the [crop filter](#cropping) overwrites point
records which are passed to it. Note that there is no undo functionality offered.

End users can read LAS/Z files, process the point clouds contained inside them, and save their work to disk at any time.
In addition, derivative terrain representations, such as a digital terrain model (DTM) and isolines, can be generated
and exported to separate files on demand, given the current point cloud state.

# Supported Operations

## Reading

A point cloud can be read into memory using the `PointCloud` class, which relies
on [laspy]( https://pypi.org/project/laspy/) and [lazrs]( https://pypi.org/project/lazrs/) for LAS and LAZ file support,
respectively. All relevant functionality is implemented as an instance method of this class.

```python
pt_cloud = point_cloud.PointCloud("example.las")
```

The `pt_cloud` object has a `data` attribute of
type [`LasData`]( https://laspy.readthedocs.io/en/latest/complete_tutorial.html) and offers the following functionality:

- Cropping
- Ground filtering
- Thinning
- DTM generation
- Saving

## Cropping

Once read, the point cloud can be cropped to the extents of an axis-aligned rectangle.

```python
# This is the coordinate pair representing the lower left (i.e., southwest) vertex of the rectangle.
x = 885264  # This is a random value.
y = 745701  # This is a random value.
pt_cloud.crop(geom.BoundingBox(geom.Point(x, y), 500, 500))
```

When executed, the above statements result in the point cloud being cropped to a 500x500 unit region, originating at the
point (885264, 745701) and extending along the positive X- and Y- semi-axes. Since the resulting region of interest is
square in shape, the third argument of the `BoundingBox` constructor, which corresponds to its length along the Y-axis,
could have been omitted without causing any unwanted behavior.

Note that the units of the rectangle are those of the coordinate reference system of the point cloud.

## Ground Filtering

```python
pt_cloud.csf()
```

There are a lot of parameteres but we spent many hours fine tuning them. Only change them if results are erroneous.

## Thinning

The point cloud can be downsampled by a given percentage using one of two supported methods. The first
is `point_cloud.DownsamplingMethod.NTH_POINT`, which operates in a similar fashion to
MATLAB’s [`downsample`](https://www.mathworks.com/help/signal/ref/downsample.html). To thin the point cloud by 50% using
n-th point sampling, the following command can be used.

```python
pt_cloud.downsample(0.5, point_cloud.DownsamplingMethod.NTH_POINT)
```

Alternatively, `point_cloud.DownsamplingMethod.RANDOM` randomly draws a certain number of points from a uniform
distribution without replacement until the end user’s requirements are satisfied. To thin the point cloud by 90% using
random point sampling, the following command can be used.

```python
pt_cloud.downsample(0.1, point_cloud.DownsamplingMethod.RANDOM)
```

Note that nth-point sampling is generally significantly faster than its random counterpart, but produces exact results
only when the quantity `percentage ** -1` evaluates to an integer. In fact, using the previous corresponding command as
an example, any percentage value from 0.4 up to 0.5 would invoke the exact same behavior from `downsample`. It is for
this reason that this sampling method should be used only under very specific circumstances. Otherwise, random sampling
is notably more versatile, and is thus used if the sampling method is not specified.

## Digital Terrain Model Generation

```python
resolution = 0.5
pt_cloud.dtm("example_dtm.tif", resolution=resolution)
```

Two methods are offered, the default is point_cloud.DTMGenerationMethod.CLASSIFICATION, the other is .CLOTH. If the dtm
is wrong to due a suspected csf issue, you need to change the default parameters of PointCloud.csf directly in the
source code. PointCloud.dtm does not expose them to the user.

### Contour line Extraction

Go to contour.py, set dtm filename and the isovalues and run the file.

## Saving

The current state of the point cloud can be saved to a separate file at any time.

```python
pt_cloud.save("example_processed.las")
```

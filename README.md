# img2poly
Convert images to 2D polygon 'art'

## To Do

- [x] Get started
- [x] get N points from canny edge
- [ ] get all pixels inside a given triangle (generalize to poligons if possible/easy)
- [ ] extract average/median/other color from a given set of pixels
- [ ] implement Poisson Disk Sampling
	- http://code-spot.co.za/2010/04/07/poisson-disk-sampling-example-code/
	- http://devmag.org.za/2009/05/03/poisson-disk-sampling/
	- https://www.jasondavies.com/poisson-disc/
- [x] create delaunay (check scipy built in function) triangulation from a given set of points
	- [ ] ?force min angle on delaunay
- [ ] switch edge detection to scikit-image (instead of opencv)
- [ ] tune edge detection
- [ ] color average: sqrt(avg(RGBs^2)) or Lab color space
- [ ] export png and svg result

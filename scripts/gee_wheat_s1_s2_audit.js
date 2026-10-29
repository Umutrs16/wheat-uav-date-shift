// Inventory Sentinel-2 L2A and Sentinel-1 GRD over the Gaocheng wheat field.
// The rectangle is an inventory window, not an official field boundary.
// Usable S2: SCL in {4, 5, 7} covers at least 80% of the window.
// No scene is treated as a same-day match unless its acquisition date is that day.

var center = ee.Geometry.Point([114.853997, 37.881194]);
var roi = center.buffer(250).bounds();
var start = '2023-10-01';
var end = '2024-06-16';

var s2 = ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
  .filterBounds(roi)
  .filterDate(start, end);

function s2Feature(img) {
  var scl = img.select('SCL');
  var clear = scl.eq(4).or(scl.eq(5)).or(scl.eq(7)).rename('clear');
  var stats = clear.reduceRegion({
    reducer: ee.Reducer.mean(),
    geometry: roi,
    scale: 20,
    maxPixels: 1e7
  });
  return ee.Feature(null, {
    scene_id: img.get('PRODUCT_ID'),
    date: img.date().format('YYYY-MM-dd'),
    tile: img.get('MGRS_TILE'),
    cloud_pct: img.get('CLOUDY_PIXEL_PERCENTAGE'),
    roi_clear: stats.get('clear')
  });
}

var s1 = ee.ImageCollection('COPERNICUS/S1_GRD')
  .filterBounds(roi)
  .filterDate(start, end)
  .filter(ee.Filter.eq('instrumentMode', 'IW'))
  .filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VV'))
  .filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VH'));

function s1Feature(img) {
  return ee.Feature(null, {
    scene_id: img.get('system:index'),
    date: img.date().format('YYYY-MM-dd'),
    orbit_pass: img.get('orbitProperties_pass'),
    relative_orbit: img.get('relativeOrbitNumber_start'),
    platform: img.get('platform_number')
  });
}

print('S2 inventory', s2.map(s2Feature));
print('S1 inventory', s1.map(s1Feature));
Map.centerObject(roi, 14);
Map.addLayer(roi, {color: 'red'}, 'inventory ROI');

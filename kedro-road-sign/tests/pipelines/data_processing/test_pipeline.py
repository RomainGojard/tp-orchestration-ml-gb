"""
This is a boilerplate test file for pipeline 'import_data'
generated using Kedro 0.19.11.
Please add your pipeline tests here.

Kedro recommends using `pytest` framework, more info about it can be found
in the official documentation:
https://docs.pytest.org/en/latest/getting-started.html
"""
from src.kedro_road_sign.pipelines.data_processing import convert_to_yolo_format
def test_convert_to_yolo_format_basic(self):
  row = {
    'Roi.X1': 50,
    'Roi.X2': 150,
    'Roi.Y1': 30,
    'Roi.Y2': 130,
    'ClassId': 2
  }
  img_w = 200
  img_h = 200
  result = convert_to_yolo_format(row, img_w, img_h)
  expected = "2 0.500000 0.400000 0.500000 0.500000"# Calcul manuel vérifié
  self.assertEqual(result, expected)

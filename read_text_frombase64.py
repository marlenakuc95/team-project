import json
import pathlib
import base64
root = pathlib.Path(__file__).parent

data_path = root.joinpath("basf_test_data")

# put a sample input path here
path = ""

with open(path) as f:
  data = json.load(f)

text = str(data['text'])
print(text)
decode = base64.b64decode(text).decode("UTF-16BE")

print(decode)
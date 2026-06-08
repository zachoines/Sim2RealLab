"""
Test file to verify type stubs are working for IntelliSense.

After generating stubs and reloading VS Code, use this file to test:
1. Autocomplete - Type a class name followed by '.' and wait for suggestions
2. Signature Help - Type a method name followed by '(' to see parameters
3. Go to Definition - Right-click on a class/method and select "Go to Definition"
"""

from pxr import Usd, UsdGeom, Gf

# Test 1: Stage methods should show autocomplete
stage = Usd.Stage.CreateInMemory()

# Try typing:  stage.Get
# You should see: GetDefaultPrim, GetPrimAtPath, GetRootLayer, etc.

# Test 2: Prim methods should show autocomplete
prim = stage.DefinePrim("/TestPrim")

# Try typing:  prim.Get
# You should see: GetAttribute, GetChildren, GetPath, etc.

# Test 3: UsdGeom classes
xform = UsdGeom.Xform.Define(stage, "/Xform")

# Try typing:  xform.
# You should see various UsdGeom.Xform methods

# Test 4: Gf classes (math types)
vec = Gf.Vec3d(1.0, 2.0, 3.0)

# Try typing:  vec.
# You should see vector methods like GetLength, GetNormalized, etc.

# Test 5: Go to Definition
# Right-click on 'Stage' above and select "Go to Definition"
# It should take you to the generated stub file

print("Type stub test file loaded!")
print("Try autocomplete on the objects above to verify stubs are working.")

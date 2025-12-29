# Import mecanum robot into Isaac Sim
- Use mechanum_robot.fbx. Based off raw [Gobilda Strafer Shassis kit stl file](https://www.gobilda.com/strafer-chassis-kit-v4/?srsltid=AfmBOopvDuIe7LDz2yBKtlYDz-er1V4DDK-8q7ytZd1nMT76AIa1XMPY). Just deleted unnessary meshes and renamed some components for identification, then exported to fbx
- Alternatively, you can directly convert using IsaacLab tool.
python C:\Worspace\IsaacLab\scripts\tools\convert_mesh.py C:\Worspace\mechanum_robot.obj C:\Worspace\mechanum_robot_obj.usd --make-instanceable

# Clean up hierarchy on mecanum robot
- python Scripts/repack_rollers.py --stage "C:/Worspace/Assets/mechanum_robot_v1.usd" --out "C:/Worspace/Assets/mechanum_robot_v1_repacked.usd"

# Generate correctly placed and oriented frames for revolute joints
- python Scripts\fix_roller_axes.py --stage "C:/Worspace/Assets/mechanum_robot_v1_repacked.usd" --out "C:/Worspace/Assets/mechanum_robot_v1_jointframes.usd"

# Add articulations
- python Scripts/fix_articulations.py --stage "C:/Worspace/Assets/mechanum_robot_v1_jointframes.usd" --out "C:/Worspace/Assets/mechanum_robot_v1_articulated.usd"

# Add collision meshes
- python Scripts/add_convex_collision.py --stage "C:/Worspace/Assets/mechanum_robot_v1_articulated.usd" --out "C:/Worspace/Assets/mechanum_robot_v1_collision.usd"

# Final processing
python Scripts/final_processing.py --stage "C:/Worspace/Assets/mechanum_robot_v1_collision.usd" --out "C:/Worspace/Assets/mechanum_robot_v1_final.usd"

# Setup test stage
- python Scripts/setup_test_stage.py --stage "C:/Worspace/Assets/mechanum_robot_v1_final.usd" --out "C:/Worspace/Assets/mechanum_robot_v1_final_test.usd"

# Other notes for converting mesh
- Verify each step with `scripts/run_empty_lab.py`     



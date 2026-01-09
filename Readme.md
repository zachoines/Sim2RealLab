# Import mecanum robot into Isaac Sim
- Use mechanum_robot.fbx. Based off raw [Gobilda Strafer Shassis kit stl file](https://www.gobilda.com/strafer-chassis-kit-v4/?srsltid=AfmBOopvDuIe7LDz2yBKtlYDz-er1V4DDK-8q7ytZd1nMT76AIa1XMPY). Just deleted unnessary meshes and renamed some components for identification, then exported to fbx
- Alternatively, you can directly convert using IsaacLab tool.
python C:\Worspace\IsaacLab\scripts\tools\convert_mesh.py C:\Worspace\mechanum_robot.obj C:\Worspace\mechanum_robot_obj.usd --make-instanceable

# Consolidate redundant Xforms when importing
- python Scripts/collapse_redundant_xforms.py --stage Assets/3209-0001-0006-v6/3209-0001-0006.usd --root /World/strafer --output ./collapse_log.txt --tree-output ./collapsed_tree.txt --output-usd ./Assets/3209-0001-0006-v6/3209-0001-0006-collapsed.usd

# Rig robot with articulations and collisions
- python Scripts/setup_physics.py --stage Assets/3209-0001-0006-v6/3209-0001-0006-collapsed.usd --output-usd Assets/3209-0001-0006-v6/3209-0001-0006-physics.usd --log ./setup_physics_log.txt --delete-excluded
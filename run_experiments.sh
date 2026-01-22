#!/bin/zsh

# Keypoints in handle
# 1. Mug 15
python -m daily_object.ply_generator --num_views 10 --is_target --object mug --object_idx 15
python -m semantic_match.semantic_s2t_ml

# 2. Mug 8
python -m daily_object.ply_generator --num_views 10 --is_target --object mug --object_idx 8
python -m semantic_match.semantic_s2t_ml

# 3. Pitcher 1
python -m daily_object.ply_generator --num_views 10 --is_target --object "pitcher_(vessel_for_liquid)" --object_idx 1
python -m semantic_match.semantic_s2t_ml

# 4. Coffeepot 0
python -m daily_object.ply_generator --num_views 10 --is_target --object coffeepot --object_idx 0
python -m semantic_match.semantic_s2t_ml

# 5. Kettle 29
python -m daily_object.ply_generator --num_views 10 --is_target --object kettle --object_idx 29
python -m semantic_match.semantic_s2t_ml

# 6. Basket 5
python -m daily_object.ply_generator --num_views 10 --is_target --object basket --object_idx 5
python -m semantic_match.semantic_s2t_ml

# 7. Basket 20
python -m daily_object.ply_generator --num_views 10 --is_target --object basket --object_idx 20
python -m semantic_match.semantic_s2t_ml

# 8. Pot 25
python -m daily_object.ply_generator --num_views 10 --is_target --object pot --object_idx 25
python -m semantic_match.semantic_s2t_ml 

# 9. Handbag 0
python -m daily_object.ply_generator --num_views 10 --is_target --object handbag --object_idx 0
python -m semantic_match.semantic_s2t_ml

# Keypoint in rim
# 1. Bowl 5
# python -m daily_object.ply_generator --num_views 10 --is_target --object bowl --object_idx 5
# python -m semantic_match.semantic_s2t_ml

# # 2. Basket 5
# python -m daily_object.ply_generator --num_views 10 --is_target --object basket --object_idx 5
# python -m semantic_match.semantic_s2t_ml

# # 3. Basket 20
# python -m daily_object.ply_generator --num_views 10 --is_target --object basket --object_idx 20
# python -m semantic_match.semantic_s2t_ml

# # 4. Pot 25
# python -m daily_object.ply_generator --num_views 10 --is_target --object pot --object_idx 25
# python -m semantic_match.semantic_s2t_ml


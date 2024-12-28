size=['n','s','m','l','x']
result=[f"/home/jzs/cv/ultralytics/voc_fp32_yolov8{i}.txt" for i in size]


import os  
  
output_file = 'merged_output.txt'  # 合并后的输出文件名  

  
# 确保输出文件不会覆盖现有文件（如果需要）  
if os.path.exists(output_file):  
    os.remove(output_file)  
  
# 写入header到输出文件  
with open(output_file, 'w') as outfile:  
    for txt_file in result:  
        with open(txt_file, 'r') as infile:  
            infile.readline()
            outfile.write(infile.readline())  # 写入第二行（可能是数据的第一行）  
  
print(f"合并完成，结果已保存到 {output_file}")

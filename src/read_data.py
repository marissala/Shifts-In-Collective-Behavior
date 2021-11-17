"""Function to read in the data, makes small samples of the datasets to work with for code dev
Author: Maris Sala
Date: 8th Nov
"""
import re
import json
import ast
import glob
import pandas as pd
import pprint
from icecream import ic

# dk_groups_clean.json  dk_groups_comments_clean.json  dk_groups_posts_clean.json

#filenames = glob.glob(data_path)
#pattern = r'}(, )'



def read_data(filename):
    with open(filename) as f:
        #lines = [next(f) for x in range(1000000)]
        #lines = f.read().splitlines("}(, ){") #f.readlines()
        giant_string = f.read()
    giant_string = giant_string.replace('\n', '\\n') #fix 1897017
    giant_string = giant_string.replace('}\s+{\s+"id"' , '}, \\n{ "id"') #fix 41432
    giant_string = giant_string.replace(' \\n}, { "id"', '}, \\n{ "id"') #fix 186777
    giant_string = giant_string.replace('\s+},\s+{\s+"id"', '}, \\n{ "id"') #fix 186803
    #fix 430987  }, {  "id"
    giant_string = giant_string.replace('"application":null', '"application":"null"')
    giant_string = giant_string.replace('}, \\n{ "id"', '}, splitstringhere\\n{ "id"') #if just this then we get 512763 lines, with the upper one it has 512763 lines, same same
    pre_lines = giant_string.split('splitstringhere\\n')
    del giant_string
    lines = [re.sub("\\\\n", " ", line) for line in pre_lines]#[line.strip('\\\\n') for line in pre_lines]
    #lines = [re.sub("\\n", " ", line) for line in lines1]
    del pre_lines
    print(len(lines))
    print(type(lines))
    print(repr(lines[0]))
    
    #pattern = r" },"
    #lines2 = [re.sub(" },", "}", line) for line in lines]
    #lines21 = [re.sub(':null', ':"null"', line) for line in lines]
    #lines22 = [re.sub('} {', '}, {', line) for line in lines21]
    #lines2 = [re.sub('" \n', '"}', line) for line in lines22]
    #lines22 = [re.sub('^}, {', '{', line) for line in lines2]
    #lines21 = [re.sub('\n', '', line) for line in lines22]
    #print(lines21[0])
    #ic(type(lines21[0]))

    i = 0
    mistakes = 0
    lines3 = []
    for line in lines:
        try:
            lines3.append(ast.literal_eval(line))
            i = i+1
        except:
            print("error in line: ", i)
            print(repr(line[0:10000]))
            #nextline=i+1
            #print("-----------------------------")
            #print("next line: ")
            #print(repr(lines[nextline][0:5000]))
            
            #print("-----------------------------")
            #print("previous line: ")
            #previousline=i-1
            #print(repr(lines[previousline][0:5000]))
            print(i)
            i = i+1
            mistakes = mistakes+1
            if mistakes > 10:
                print(maris)
            else:
                continue
            
    del lines
    print(f"NUMBER OF MISTAKES: {mistakes}")
    #lines3 = [ast.literal_eval(line) for line in lines2]
    df = pd.DataFrame.from_records(lines3)#, index=True)
    del lines3
    print(f"LENGTH OF DF: ", len(df))
    return df

"""
for filename in filenames:
    ic(filename)
    with open(filename) as f:
        lines = f.read().splitlines() #f.readlines()
    ic(filename)
    ic(len(lines))
    hmm = re.sub(pattern, "}", lines[0])
    hmm2 = ast.literal_eval(hmm)
    df = pd.DataFrame.from_records([hmm2])#, index=True)
    ic(df.columns)
"""

if __name__ == '__main__':
    ic("START PIPELINE---------------------------------")
    data_path = "/data/datalab/danish-facebook-groups/raw-export/*"
    filenames = glob.glob(data_path)
    df = read_data(filenames[2])
    #dff = df[:100]
    #dff.to_csv("/home/commando/marislab/facebook-posts/tmp/tmp_df.csv", index = False, sep=";")
    ic(df.head())
    ic("ALL DONE---------------------------------------")
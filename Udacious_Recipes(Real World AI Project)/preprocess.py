import os
import pandas as pd
import re
import csv
import zipfile
import json
import itertools
import numpy as np
import linecache
import sys


def merge_all_cats(all_cats_dict): #merges all categories into single lists and their associated labels
    
    titles=all_cats_dict['titles']
    drns=all_cats_dict['directions']
    ingrs=all_cats_dict['ingredients']
    junks=all_cats_dict['junks']
    
    titles_lables=["title"]*len(titles)
    dirns_lables=["direction"]*len(drns)
    ingrs_lables=["ingredient"]*len(ingrs)
    junks_lables=["junk"]*len(junks)
    
    all_examples=merge_lists([titles,drns,ingrs,junks])
    all_labels=merge_lists([titles_lables,dirns_lables,ingrs_lables,junks_lables])
    
    return pd.DataFrame(data=np.column_stack([all_examples,all_labels]),columns=["Examples","Labels"])

def merge_lists(lists):
        
        flat_list=[item for sublist in lists for item in sublist]
        
        return flat_list
    

################################ preprocessing fucntions for recipes-raw dataset ##########################################

def get_recipes_raw(folder_path):
    
    dirs_list=[]
    ingrs_list=[]
    titles_list=[]
    junks_list=[]
    
    json_files=os.listdir(folder_path)
    ex=0
    
    for json_file in json_files:

        json_file_path=folder_path+"/"+json_file

        if os.path.isfile(json_file_path) and json_file.endswith("json"):
           
                with open(json_file_path,'r') as f:

                    file_content_dict=json.load(f)
                    
                    for info in file_content_dict.items():
                        
                        try:
                        
                            key=info[0]
                            file_content=info[1]
                            titles_list.append(file_content["title"])
                            dirs_list.append(file_content["instructions"])
                            ingrs=file_content["ingredients"]
                            ingrs_list+=ingrs
                           
                        except Exception as e:
                            pass
                            #ex+=1
                            #print (key,json_file_path,file_content)
                            #PrintException()

    #print(ex)
    return titles_list,dirs_list,ingrs_list,junks_list        
######################################################################################################################################

################################ preprocessing fucntions for recipes-master dataset ##########################################

def get_recipes_master(folder_path):
    
    dirs_list=[]
    ingrs_list=[]
    titles_list=[]
    junks_list=[]
    
    all_folders=os.listdir(folder_path)
   
    for folder in all_folders:
        dir_path=folder_path+"/"+folder
        
        if os.path.isdir(dir_path):
            
            json_files=os.listdir(dir_path)
            
            for json_file in json_files:
                
                json_file_path=folder_path+"/"+folder+"/"+json_file
               
                if os.path.isfile(json_file_path) and json_file.endswith("json"):
                    
                    try:
                        with open(json_file_path,'r') as f:
                            
                            file_content=json.load(f)

                            #print("File Reading Success : {}".format(json_file_path))
                            #valid file contents
                            dirns=file_content["directions"]
                            title=file_content["title"]
                            ingrs=file_content["ingredients"]

                            #junk contents

                            url=file_content["url"]
                            tags=file_content["tags"]
                            source=file_content["source"]

                            #add all to relevant lists
                            dirs_list+=dirns
                            titles_list.append(title)
                            ingrs_list+=ingrs

                            junks_list.append(source)
                            junks_list.append(url)
                            junks_list+=tags
                    
                    except Exception as e:
                        print("File Reading Failure : {}".format(json_file_path))

                        
    return titles_list,dirs_list,ingrs_list,junks_list        




#############################################################################################################################



##################################### preprocessing functions for reading csv files data ################################

def unzip_files(dir_source,dir_dest):
    
    for file_name in os.listdir(dir_source):
        
        if file_name.endswith("zip"):
            zip_path=dir_source+"/"+file_name
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                print("Unzipping {} ... ".format(file_name))
                zip_ref.extractall(dir_dest)

def get_dirns_from_string(dirn_str):
    
    dirn_out=""
    
    if isinstance(dirn_str,str):
        dirn_out=dirn_str.split("\r\n")
    
    return dirn_out

def str_to_list(list_str):
    
    list_str=re.sub('\'','"',re.sub('(\[|\])',"",list_str))#.strip('"')
    
    if isinstance(list_str,str):
        
        list_str=re.findall('".*?"',list_str)
        
    else:list_str=""   
    
    return list_str



def fmt_df(list_df):  # formats list of dataframes into 4 - title, ingredients, directions, junk,
    
    titles=[]
    dirs=[]
    junks=[]
    ingrs=[]
    
    for df in list_df:
        
        titles_df=df[0].tolist()
        ingrs_df=df[1].apply(str_to_list)
        dirs_df=df[2].apply(get_dirns_from_string)
        junks_df=df[3].apply(str_to_list)
        
        #for t in titles_df: print(t)
        """print()
        for i in ingrs_df : print(i)
        print()
        for d in dirs_df: print(d)
        print()
        for j in junks_df: print(j)"""
        
        titles+=titles_df
        
        ingrs+=ingrs_df
        junks+=junks_df
        dirs+=dirs_df

    
    return titles,dirs,junks,ingrs    

def read_files_csv(data_folder="./datasets/csv_files"): #processes csv_files folder of dataset
    
    files=os.listdir(data_folder)
    
    df_list=[]
    
    for file in files:
        
        if file.endswith("csv"):
            
            file_path=data_folder+"/"+file
            file_df=pd.read_csv(file_path,encoding='unicode_escape',header=None)
            df_list.append(file_df)
            print("File Read : {} Examples : {}".format(file_path,file_df.shape))
        
    return df_list

####################################################################################################################################

"""def PrintException():
    exc_type, exc_obj, tb = sys.exc_info()
    f = tb.tb_frame
    lineno = tb.tb_lineno
    filename = f.f_code.co_filename
    linecache.checkcache(filename)
    line = linecache.getline(filename, lineno, f.f_globals)
    print ('EXCEPTION IN ({}, LINE {} "{}"): {}'.format(filename, lineno, line.strip(), exc_obj))"""
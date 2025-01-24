import sys
from pyserini.search import FaissSearcher
from transformers import T5ForConditionalGeneration
from transformers import T5Tokenizer
import torch
import pandas as pd
import json
from tqdm import tqdm
import os
import numpy as np
import re
from transformers import DPRQuestionEncoderTokenizer
import warnings
from transformers import AutoTokenizer
from pyserini.search.lucene import LuceneSearcher
from TrueTeacher_utils import *
import socket

if __name__ == '__main__':
    cp = "comp"
    online = False
    EF_debug = False
    print(f"starting {cp} analysis...")

    if os.path.exists(f'/lv_local/home/user/llama/TT_input/feature_data_{cp}_new_F.csv'):
        df = pd.read_csv(f'/lv_local/home/user/llama/TT_input/feature_data_{cp}_new_F.csv').astype(str)

    else:
        if os.path.exists(f'/lv_local/home/user/llama/TT_input/feature_data_{cp}_new.csv'):
            df = pd.read_csv(f'/lv_local/home/user/llama/TT_input/feature_data_{cp}_new.csv').astype(str)
            df = df[df.creator != 'creator']

        else:
            df = create_feature_df(cp, online)


    eval_measures = ['CF@1', 'CF@1_ref', 'NCF@1',
                     'EF@10_dense', 'EF@10_sparse',
                     'NEF@10_dense', 'NEF@10_sparse',
                     'ref_pos_sparse', 'ref_pos_dense']

    for measure in eval_measures:
        if measure not in df.columns:
            df[measure] = "nan"

   
    if cp == 'comp':
        
        df[['round_no', 'query_id', 'username', 'creator']] = df.docno.apply(lambda x: pd.Series(x.split('-')))
        df[['round_no', 'query_id', 'username']] = df[['round_no', 'query_id', 'username']].astype(int).astype(str)

    try:
        y_cols = [col for col in df.columns if '_y' in col]
        df = df.drop(y_cols, axis=1)
    except:
        pass

    nan_indices = df.index[
        df[eval_measures].apply(lambda row: any(row == 'nan') or any(row == np.nan), axis=1)].tolist()

    if len(nan_indices) == 0:
        print("all done!")
        df.to_csv(f'/lv_local/home/user/llama/TT_input/feature_data_{cp}_new_F.csv', index=False)
        columns_to_convert = [col for col in eval_measures if col != 'CF@1_values']
        df[columns_to_convert] = df[columns_to_convert].astype(float)
        df[eval_measures] = df[eval_measures].replace('nan', np.nan)
        mean_df = df.groupby(['username'])[columns_to_convert].mean().reset_index()
        print(mean_df)
        exit()

    print("remaining: ", len(nan_indices))
    try:
        text_df = pd.read_csv(f'/lv_local/home/user/llama/TT_input/bot_followup_{cp}.csv').astype(str)
    except:
        text_df = pd.read_csv(f'/lv_local/home/user/llama/TT_input/feature_data_{cp}_new.csv').astype(str)

    if cp == 'student':
        text_df['creator'] = 'creator'

    text_df = text_df.astype(str).merge(df[['query_id', 'creator', 'username', 'docno']].astype(str),
                                        on=['query_id', 'creator', 'username'],
                                        how='left', suffixes=('', '_y'))
    if text_df.docno.isna().all():
        text_df.drop('docno', axis=1, inplace=True)
        if "temp" in text_df.columns:
            text_df['username'] = text_df['username'].astype(str) + "@" + text_df['temp'].apply(
                lambda x: str(int(float(x) * 100)))  # ***
        text_df = text_df.astype(str).merge(df[['query_id', 'creator', 'username', 'docno']].astype(str),
                                            on=['query_id', 'creator', 'username'],
                                            how='left', suffixes=('', '_y'))

    if not online:
        g_df = pd.read_csv(f'/lv_local/home/user/llama/g_data.csv').astype(str)
    else:
        g_df = pd.read_csv(f'/lv_local/home/user/llama/full_online_data.csv').astype(str)
    g_df['new_docno'] = g_df.apply(lambda row: row.docno.split('ROUND-')[1].replace("00", "0") + "-creator",
                                         axis=1)



    counter = 0
    error_counter = 0

    cache = load_cache()
    cache_size = get_cache_size(cache)

    for idx, row in df.iterrows():
        if row.previous_docno_str not in data_dict.keys():  
            df.at[idx, 'previous_docno_str'] = re.sub(r"-(\d)-", lambda m: f"-0{m.group(1)}-", row.previous_docno_str)
            assert df.at[idx, 'previous_docno_str'] in names_nto


    for idx, row in tqdm(df.iterrows(), total=df.shape[0]):
        if row.round_no == '1':
            continue
        if idx not in nan_indices:
            continue


        try:
            k = 10
            try:
                text = text_df[text_df.docno == row.docno].text.values[0]  
            except:
                try:
                    text = g_df[g_df.new_docno == row.docno.replace("-" + row.docno.split("-")[1] + "-",
                                                                              "-" + row.docno.split("-")[1].zfill(
                                                                                  3) + "-").replace("00","0")].current_document.values[0]
                except:
                    text = g_df[g_df.new_docno ==  re.sub(r"-(\d)-", lambda m: f"-0{m.group(1)}-", row.docno)].current_document.values[0]


            sentences = [sentence.strip() for sentence in re.split(r'[.!?]', text) if sentence]
            sentences_duo = [sentences[i] + '. ' + sentences[i + 1] for i in range(len(sentences) - 1)]

            text = truncate_to_max_tokens(text, max_tokens=512)
            try:
                if 'creator' in row.docno or row.creator == 'creator':  
                    try: 
                        prev_text = g_df[g_df.docno == row.docno.replace(f"0{int(row.round_no)}",
                                                                               f"0{int(row.round_no) - 1}")].current_document.values[
                            0]
                    except:
                        prev_text = g_df[g_df.new_docno == row.docno.replace(f"0{int(row.round_no)}",
                                                                                   f"0{int(row.round_no) - 1}").replace(
                            "-" + row.docno.split("-")[2] + "-",
                            "-0" + row.docno.split("-")[2] + "-")].current_document.values[0]
                else:
                    prev_text = g_df[g_df.new_docno == row.previous_docno_str].current_document.values[0]
            except:
                if 'previous_docno_str' in df.columns and row.previous_docno_str != 'nan':
                    try:
                        if online:
                            prev_text = g_df[g_df.docno == row.previous_docno_str].current_document.values[0]
                        else:
                            prev_text = g_df[g_df.docno == row.previous_docno_str].current_document.values[0]
                    except:
                        prev_text = g_df[g_df.new_docno == row.previous_docno_str.replace(
                            "-" + row.previous_docno_str.split("-")[2] + "-",
                            "-0" + row.previous_docno_str.split("-")[2] + "-")].current_document.values[0]


            try: 
                prev_text = data_dict[row.previous_docno_str]
            except:
                prev_text = data_dict[names_nto[row.previous_docno_str]]
            prev_text = truncate_to_max_tokens(prev_text, max_tokens=512)

            with torch.no_grad():
                if row['NEF@10_dense'] == 'nan' or row['NEF@10_max_dense'] == 'nan' or row['EF@10_dense'] == 'nan' or \
                        row['EF@10_max_dense'] == 'nan' or row['NEF@5_dense'] == 'nan' or row[
                    'NEF@5_max_dense'] == 'nan' or row['EF@5_dense'] == 'nan' or row['EF@5_max_dense'] == 'nan' or \
                        row['NEF@10_sparse'] == 'nan' or row['NEF@10_max_sparse'] == 'nan' or row[
                    'EF@10_sparse'] == 'nan' or row['EF@10_max_sparse'] == 'nan' or row['NEF@5_sparse'] == 'nan' or row[
                    'NEF@5_max_sparse'] == 'nan' or row['EF@5_sparse'] == 'nan' or row[
                    'EF@5_max_sparse'] == 'nan' or EF_debug:
                   

                    top_docs_dense_docnos, top_docs_sparse_docnos, _, _ = retrieve_top_docs(text, k=k,
                                                                                            round_no=row.round_no,
                                                                                            online=online)

                    metrics_dense = process_top_docs(top_docs_dense_docnos, g_df, sentences, k, cache, online=online)
                    metrics_sparse = process_top_docs(top_docs_sparse_docnos, g_df, sentences, k, cache, online=online)

                    store_results(df, idx, metrics_dense, "dense", k, k // 2)
                    store_results(df, idx, metrics_sparse, "sparse", k, k // 2)

                if row['ref_pos_sparse'] == 'nan' or row['ref_pos_dense'] == 'nan' or EF_debug:
                    top_docs_dense_docnos, top_docs_sparse_docnos, _, _ = retrieve_top_docs(text, k=k,
                                                                                            round_no=row.round_no,
                                                                                            online=online)
                    df.loc[idx, 'ref_pos_sparse'] = (
                        top_docs_sparse_docnos.index(row.previous_docno_str) + 1
                        if row.previous_docno_str in top_docs_sparse_docnos
                        else (
                            top_docs_sparse_docnos.index(names_nto[row.previous_docno_str]) + 1
                            if row.previous_docno_str in names_nto and names_nto[
                                row.previous_docno_str] in top_docs_sparse_docnos
                            else 999
                        )
                    )

                    df.loc[idx, 'ref_pos_dense'] = (
                        top_docs_dense_docnos.index(row.previous_docno_str) + 1
                        if row.previous_docno_str in top_docs_dense_docnos
                        else (
                            top_docs_dense_docnos.index(names_nto[row.previous_docno_str]) + 1
                            if row.previous_docno_str in names_nto and names_nto[
                                row.previous_docno_str] in top_docs_dense_docnos
                            else 999
                        )
                    )
                    x = 1
                if row['CF@1'] == 'nan' or row['NCF@1'] == 'nan' or EF_debug:
                    cf_vals, cf_ref_vals, norm_vals = calculate_metrics([prev_text], sentences, cache)
                    df.loc[idx, 'CF@1'] = cf_vals[0]
                    df.loc[idx, 'CF@1_ref'] = cf_ref_vals[0]
                    df.loc[idx, 'NCF@1'] = norm_vals[0]



                if cache_size != get_cache_size(cache):
                    save_cache(cache)
                    cache_size = get_cache_size(cache)

                for suffix in ['dense', 'sparse']:
                    if df.loc[idx, f'ref_pos_{suffix}'] != 999:
                        nef_value = df.loc[idx, f'NEF@{k}_max_{suffix}']
                        ncf_value = df.loc[idx, 'NCF@1']
                        ef_value = df.loc[idx, f'EF@{k}_max_{suffix}']
                        cf_value = df.loc[idx, 'CF@1']

                        if float(nef_value) < float(ncf_value):
                            print(
                                f"NEF@{k}_max_{suffix} (value: {nef_value}) is less than NCF@1 (value: {ncf_value}) for index {idx}")

                        if float(ef_value) < float(cf_value):
                            print(
                                f"EF@{k}_max_{suffix} (value: {ef_value}) is less than CF@1 (value: {cf_value}) for index {idx}")



                counter += 1
                if counter % 10 == 0 or counter == len(nan_indices) - error_counter:
                    df.to_csv(f'/lv_local/home/user/llama/TT_input/feature_data_{cp}_new_F.csv', index=False)
                    df[eval_measures] = df[eval_measures].replace('nan', np.nan)
                    columns_to_convert = [col for col in eval_measures if col != 'CF@1_values']
                    df[columns_to_convert] = df[columns_to_convert].astype(float)
                    mean_df = df.groupby(['username'])[columns_to_convert].mean().reset_index()

                    if counter == len(nan_indices) - error_counter:
                        save_cache(cache)


        except Exception as e:
            exc_type, exc_obj, exc_tb = sys.exc_info()
            print(exc_type, exc_tb.tb_lineno, e)
            error_counter += 1
            continue

import pandas as pd
import re 
import string 

###### [TEXT OPERATORS] ################################################################################################

PUNCTUATION_PATTERN = f"[{re.escape(string.punctuation)}\n\t\r]"
EMAIL_PATTERN = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,7}\b'
# r"([a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+)"
URL_PATTERN = r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+|www\.[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}|[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}'
DATE_PATTERNS = [r'^\d{4}-\d{2}-\d{2}$',
                 r'^\d{2}/\d{2}/\d{4}$',
                 r'^\d{2}-\d{2}-\d{4}$', 
                 r'^\d{2}-\d{2}-\d{2}$',
                 r'^\d{1,2}(st|nd|rd|th)?\s+[A-Za-z]+\s+\d{4}$',  
                 r'^[A-Za-z]+\s+\d{1,2},\s+\d{4}$']

                # 17-May-23 / Fri May  5 20:33:00 2023

DATE_PATTERN = '|'.join(DATE_PATTERNS)

def replace_url_with_token(text:str) -> str:
    clean_text =  re.sub(URL_PATTERN, '[url]', text)
    return clean_text

def replace_email_address_with_token(text:str) -> str:
    clean_text =  re.sub(EMAIL_PATTERN, '[email_address]', text)
    return clean_text

def replace_date_with_token(text:str) -> str:
    clean_text =  re.sub(DATE_PATTERN, '[date]', text)
    return clean_text

#TODO
def remove_email_formatting(text:str) -> str:
    """ Should ignore maybe email formattingthat can create difficulty [maybe replace with token of  [email_formaating]"""
    pass #use numbers to mark different emails ? [FLAG]

def remove_punctuation_in_text_lower(text:str) -> str:
    """
    Removing punctuation and new-line,tab & \r formatting to create clean text to compare and remove punctuation duplicates
    + remove excessive whitespace post punctuation
    """
    cleaned_text = re.sub(PUNCTUATION_PATTERN, ' ', text)
    cleaned_text = re.sub(r'\s+', ' ', cleaned_text).strip()
    cleaned_text = ' '.join(cleaned_text.split())
    return cleaned_text.lower()
    
def is_text_only_digits(text:str, thresh:float) -> bool:
    """
    Should run after removing punctuation and other formatting
    """
    digit_count = sum(char.isdigit() for char in text)
    non_digit_non_space_count = sum(not char.isspace() and not char.isdigit() for char in text)
    if non_digit_non_space_count > 0:
        ratio = digit_count / non_digit_non_space_count
    else:
        ratio = 1
    
    if ratio > thresh:
        return True
    else:
        return False

###### [DF DUPLICATE OPERATORS] ##########################################################################################

def drop_punctuation_duplicates(df,text_column:str) -> object:
    """ this function returns a copy of  a dataframe without duplicates based on punctuation"""
    altered_df = df.copy()
    # altered_df['punctuation_cleaned_text'] = altered_df[text_column].apply(lambda x: remove_punctuation_in_text(x))
    altered_df.drop_duplicates(subset = [text_column], inplace = True, key = lambda x: remove_punctuation_in_text_lower(x))
    print(f'Original Dataframe size: {str(len(df))}')
    print(f'Post dropping punctuation duplicates, Dataframe size: {str(len(altered_df))}')
    # altered_df.drop(columns = ['punctuation_cleaned_text'], inplace = True)
    return altered_df
    
def mark_text_is_duplicate_based_punctuation(df, text_column:str, 
                                             duplicate_marker_column:str = 'duplicate_of_index',
                                             duplicates_text_column:str = 'duplicate_of_text',
                                             duplicate_keep:str = 'first') -> object:
    """
    This function returns *the original df passed in the function* with an additional column that marks the index of the duplicate text
    based on punctuation.
    """
    df[duplicate_marker_column] = None
    df[duplicates_text_column] = None
    df['punctuation_cleaned_text'] = df[text_column].apply(lambda x: remove_punctuation_in_text_lower(x))
    duplicates = df.duplicated('punctuation_cleaned_text', keep = duplicate_keep)
    
    for idx in df[duplicates].index:
        duplicate_value = df.iloc[idx]['punctuation_cleaned_text']
        original_idx = df[(df['punctuation_cleaned_text'] == duplicate_value)].index[0]
        if not pd.isnull(original_idx):
            df.at[idx, duplicate_marker_column] = original_idx
            df.at[idx, duplicates_text_column] = df.at[original_idx, text_column]

    df.drop(columns = ['punctuation_cleaned_text'], inplace = True)
    return df 

def extract_textual_duplicates(df, text_column, duplicate_marker_columns = 'duplicate_of_index') -> object:
    """
    This function returns a dataframe with only the duplicates based on punctuation grouped by the index of the original text
    """
    df_duplicates = df.dropna(subset=[duplicate_marker_columns])
    grouped_by_frames = [x for _, x in df_duplicates.groupby([duplicate_marker_columns])]
    return pd.concat(grouped_by_frames)

###### [DF OPERATORS] ################################################################################################

def complete_process_text(text:str) -> str:
    text = replace_email_address_with_token(text)
    text = replace_date_with_token(text)
    text = replace_url_with_token(text)
    return text

def process_textual_column_in_dataframe(df, text_column:str = 'text',
                                        remove_digit_only:bool = True, clean_text:bool = True,
                                        drop_or_mark_duplicates:str = 'drop'):
    if remove_digit_only:
        df['text_is_digits'] = df[text_column].apply(lambda x: is_text_only_digits(x, 0.8))
        df = df[~df['text_is_digits']]
        df.drop(columns = ['text_is_digits'], inplace = True)
    if clean_text:
        df['clean_text'] = df[text_column].apply(lambda x: complete_process_text(x))
        text_column = 'clean_text'
    
    if drop_or_mark_duplicates == 'drop':
        return drop_punctuation_duplicates(df, text_column)
    elif drop_or_mark_duplicates == 'mark':
        return mark_text_is_duplicate_based_punctuation(df, text_column)

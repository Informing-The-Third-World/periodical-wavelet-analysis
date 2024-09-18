import pandas as pd
from tqdm import tqdm
import numpy as np
import altair as alt
import re

def get_combined_data(directories):
    dfs = []
    for directory in tqdm(directories, desc="Loading Data"):
        dir_path = os.path.join("../datasets/ht_ef_datasets/", directory)
        for _, _, files in os.walk(dir_path):
            for f in files:
                if f.endswith(".csv"):
                    try:
                        df = pd.read_csv(os.path.join(dir_path, f))
                        file_name = f.split('.')[0]
                        filenames = file_name.split('_')
                        filenames = [ fi for fi in filenames if fi.isdigit() == False]

                        filenames = '_'.join(filenames)
                        df['cleaned_magazine_title'] = filenames
                        dfs.append(df)
                    except:
                        print("Error with", f)
    combined_df = pd.concat(dfs, ignore_index=True)
    combined_df.sequence = combined_df.sequence.astype(int)
    combined_df.token = combined_df.token.fillna('')
    combined_df.token = combined_df.token.astype(str)
    combined_df['count'] = combined_df['count'].astype(int)
    grouped_df = combined_df.groupby([
    'language',
    'publisher',
    'genre',
    'title',
    'pub_place',
    'source_institution',
    'cleaned_magazine_title',
    'section',
    'pub_date',
    'htid', 'sequence'], as_index = False).agg({'token': ' '.join, 'pos': list, 'count': sum, 'section': list})
    subset_df = grouped_df[grouped_df['count'] > 100]
    return subset_df


output_path = '../ht_ef_datasets/full_hathitrust_annotated_magazines_with_htids.csv'
output_directory = "../ht_ef_datasets/"
df = get_full_combined_dataset(output_path, output_directory)

df.token = df.token.astype(str)
df.volume_number = df.volume_number.fillna(0)

directories  = os.listdir("../datasets/ht_ef_datasets/")
directories = [directory for directory in directories if not directory.endswith(".csv")]
existing_data = pd.read_csv("../datasets/ht_ef_datasets/combined_full_hathitrust_annotated_magazines_with_htids.csv")
existing_titles = existing_data["cleaned_magazine_title"].unique().tolist()
directories = [directory for directory in directories if directory not in existing_titles]
# directories = ["Direct_from_Cuba_HathiTrust"]

subset_df = get_combined_data(directories)

# subset_df = df[df.htid == "uc1.l0063473003"]
# subset_df.cleaned_magazine_title = "Direct_from_Cuba"
subset_df = subset_df.rename(columns={'token': 'text', 'sequence': 'page_number' })
subset_df['cleaned_issue_date'] = subset_df.pub_date.astype(str) + '-01-01'
subset_df['cleaned_issue_date'] = pd.to_datetime(subset_df['cleaned_issue_date'], errors='coerce')


subset_digits = subset_df[(subset_df.token.str.isdigit()) ]
subset_digits['number'] = subset_digits.token.astype(int)
subset_digits['implied_zero'] = subset_digits.sequence.astype(int) - subset_digits.number
htids = subset_digits.htid.unique()
possible_pages = []
for htid in htids:
    rows = subset_digits[subset_digits.htid == htid]
    max_page = rows.sequence.max()
    max_possible_number = max_page + 25
    filtered_df = rows[rows.number < max_possible_number]
    possible_pages.append(filtered_df)
possible_pages_df = pd.concat(possible_pages)

implied_zero_df = subset_digits[(subset_digits.number < 500) & (subset_digits.implied_zero > 0)].groupby(['implied_zero']).size().reset_index(name='zero_counts')
top_zeros = implied_zero_df[implied_zero_df.zero_counts > 10]

def filter_integers(token):
    return bool(re.match(r'^\d+$', token))


possible_pages = subset_digits[subset_digits['token'].apply(filter_integers)].copy()
# Filter possible_pages to only contain integers and parse them
possible_pages['number'] = pd.to_numeric(possible_pages['token'], errors='coerce')
possible_pages = possible_pages.dropna(subset=['number'])
possible_pages['number'] = possible_pages['number'].astype(int)

# Derive implied_zero
possible_pages['implied_zero'] = possible_pages['sequence'] - possible_pages['number']

# Calculate raw_scores
max_page = possible_pages['sequence'].max()
max_possible_number = max_page + 25

raw_scores = np.zeros((max_page + 1, max_possible_number + 1), dtype=int)

for _, row in possible_pages.iterrows():
    seq, num = row['sequence'], row['number']
    if num <= max_possible_number:
        raw_scores[seq, num] += 1

# Create prefix_sums function
def prefix_sums(raw_scores, updown=0.5, diag=0.25, otherwise=0.01, points=0.2):
    scores = raw_scores.copy()
    nrows, ncols = raw_scores.shape
    for i in range(nrows):
        for j in range(ncols):
            cell = otherwise + points * raw_scores[i, j]
            choices = []
            if j > 0:
                choices.append(scores[i, j-1] * updown)
            if i > 0:
                choices.append(scores[i-1, j] * updown)
                if j > 0:
                    choices.append(scores[i-1, j-1] * diag)
            cell += max(choices, default=0)
            scores[i, j] = cell
    return scores

# Calculate forward and backward scores
forward = prefix_sums(raw_scores)
backward = prefix_sums(raw_scores[::-1, :])[::-1, :]

# Calculate predictions
backpass_weight = 0.66
combined_scores = backpass_weight * backward + (1 - backpass_weight) * forward
predictions = pd.DataFrame([(i, j) for i in range(combined_scores.shape[0]) for j in [np.argmax(combined_scores[i])]], columns=['sequence', 'predicted_page'])

# Merge predictions with possible_pages
result = pd.merge(possible_pages, predictions, on=['sequence'])

# Create visualization
chart = alt.Chart(result).mark_circle(size=100, color='crimson').encode(
    x=alt.X('sequence:Q', axis=alt.Axis(title='Sequence')),
    y=alt.Y('predicted_page:Q', axis=alt.Axis(title='Page')),
)

def run_prefix_sums(raw_scores, backwards=False, otherwise=0, points=1, updown=1, diag=1):
    scores = raw_scores
    filled = np.zeros((raw_scores.shape[0], raw_scores.shape[1]))

    for i in range(raw_scores.shape[0]):
        for j in range(raw_scores.shape[1]):
            cell = otherwise + points * raw_scores[i, j]
            choices = []

            if j > 0:
                choices.append(filled[i, j-1] * updown)

            if i > 0:
                choices.append(filled[i-1, j] * updown)

                if j > 0:
                    choices.append(filled[i-1, j-1] * diag)

            cell += max(choices, default=0)
            filled[i, j] = cell

    return filled

# Assuming 'possible_pages' is a DataFrame with columns 'sequence', 'number', 'token', and 'implied_zero'
max_page = int(possible_pages['sequence'].max())
max_possible_number = max_page + 25
raw_scores = np.zeros((max_page + 1, max_possible_number))

for index, row in possible_pages.iterrows():
    sequence, number = int(row['sequence']), int(row['number'])
    
    if number > max_possible_number:
        continue
        
    raw_scores[sequence, number] += 1

# Continue with the rest of the script
# ...

print(raw_scores)

def input_backpass_weight():
    while True:
        try:
            backpass_weight = float(input("Enter the weights of forward and backward passes (0=forward only; 1=backward only, default=0.66): "))
            if 0 <= backpass_weight <= 1:
                return backpass_weight
            else:
                print("Please enter a value between 0 and 1.")
        except ValueError:
            print("Invalid input. Please enter a valid number between 0 and 1.")

# ...

backpass_weight = input_backpass_weight()



# Assuming 'raw_scores' is a 2D NumPy array
raw_scores = np.array(raw_scores)

backward = run_prefix_sums(raw_scores[::-1])
backward = backward[::-1]

forward = run_prefix_sums(raw_scores)

# Assuming 'backpass_weight' is a scalar value
predictions = np.argmax(forward + backward * backpass_weight, axis=1)
predictions = [(sequence, page) for sequence, page in enumerate(predictions)]

print(predictions)

predictions_df = pd.DataFrame(predictions, columns=['sequence', 'predicted_page'])

# Merge 'predictions_df' with 'possible_pages' DataFrame
merged_df = pd.merge(possible_pages, predictions_df, on='sequence')

merged_df.predicted_page.value_counts()

import matplotlib.pyplot as plt

# ...

def visualize_scores(possible_pages, scores, predictions):
    max_page = possible_pages['sequence'].max()
    max_possible_number = max_page + 25

    # Create a figure and axis for the visualization
    fig, ax = plt.subplots()

    # Display the scores as an image
    ax.imshow(scores.T, cmap='gray', origin='lower', aspect='auto')

    # Draw rectangles around the predicted pages
    for sequence, page in predictions:
        rect = plt.Rectangle((sequence - 2, page - 2), 5, 5, edgecolor='crimson', facecolor='none', linewidth=1.5)
        ax.add_patch(rect)

    # Set axis labels
    ax.set_xlabel('Sequence')
    ax.set_ylabel('Page')

    # Show the visualization
    plt.show()

# ...

# Call the visualization function
visualize_scores(possible_pages, raw_scores, predictions)



def visualize_scores_altair(possible_pages, scores, predictions):
    max_page = possible_pages['sequence'].max()
    max_possible_number = max_page + 25

    # Convert scores to a DataFrame
    scores_df = pd.DataFrame(scores)
    scores_df = scores_df.stack().reset_index()
    scores_df.columns = ['sequence', 'page', 'value']

    # Convert predictions to a DataFrame
    predictions_df = pd.DataFrame(predictions, columns=['sequence', 'page'])
    predictions_df['highlight'] = True

    # Calculate the width and height of the rectangles based on the heatmap grid
    width = (scores_df['sequence'].max() - scores_df['sequence'].min()) / len(scores)
    height = (scores_df['page'].max() - scores_df['page'].min()) / len(scores[0])

    # Create the base Altair chart
    base = alt.Chart(scores_df).encode(
        x=alt.X('sequence:Q', title='Sequence', scale=alt.Scale(padding=0.5*width)),
        y=alt.Y('page:Q', title='Page', scale=alt.Scale(padding=0.5*height))
    )

    # Create a heatmap for scores
    heatmap = base.mark_rect(width=width, height=height).encode(
        color=alt.Color('value:Q', scale=alt.Scale(scheme='greys'))
    )

    # Create a layer with red rectangles for predictions
    highlight = alt.Chart(predictions_df).mark_rect(width=width+2, height=height+2, stroke='crimson', strokeWidth=2).encode(
        x='sequence:Q',
        y='page:Q',
        color=alt.value('crimson'),
        opacity=alt.value(1),
        tooltip=['sequence', 'page']
    )

    # Combine the heatmap and highlight layers
    chart = heatmap + highlight

    return chart

# ...

# Call the visualization function
chart = visualize_scores_altair(possible_pages, raw_scores, predictions)
chart

scores_matrix = Matrix(raw_scores.shape, fill=0)


scores_matrix.img(predictions=predictions)

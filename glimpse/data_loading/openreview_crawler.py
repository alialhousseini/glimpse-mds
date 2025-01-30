"""
Script used to extract reviews and metareviews from OpenReview API
and save them to a CSV file.

"""
import openreview
import csv

# Initialize the OpenReview client
client = openreview.Client(baseurl='https://api.openreview.net')

# Replace 'ICLR.cc/2023/Conference' with the ID of your desired conference
conference_id = 'ICLR.cc/2022/Conference'

# Apply filter & map functions


def get_decision(item: dict) -> str | None | Exception:
    content = item['content']
    # for 2023 'metareview:_summary,_strengths_and_weaknesses', 'justification_for_why_not_higher_score', 'justification_for_why_not_lower_score'
    if 'metareview:_summary,_strengths_and_weaknesses' in content:
        try:
            x = content['metareview:_summary,_strengths_and_weaknesses'] + '-----' + \
                content['justification_for_why_not_higher_score'] + \
                '-----' + content['justification_for_why_not_lower_score']
            return x
        except KeyError as e:
            return e

    else:
        return None


# Fetch submissions
submissions = client.get_all_notes(
    invitation=f'{conference_id}/-/Blind_Submission', details='directReplies')

# # Open a CSV file for writing
with open('all_reviews_2022.csv', mode='w', newline='', encoding='utf-8') as file:
    writer = csv.writer(file)
    # Write the header
    writer.writerow(['id', 'review', 'metareview'])

    # Iterate through submissions
    for idx, submission in enumerate(submissions):
        # get id
        id = submission.to_json()['id']
        id = 'https://openreview.net/forum?id='+str(id)

        # filter data and extract 'metareview'
        metareview = list(filter(lambda x: x is not None, map(
            lambda x: get_decision(x), submission.details['directReplies'])))[0]
        assert metareview is not None, f"Metareview not found for {id}"
        if isinstance(metareview, Exception):
            print(f'### idx {idx} ###')
            extracted_reviews.clear()
            continue
        else:
            metareview = metareview.replace('\n', '-----')
            metareview = metareview.replace('"', '""')
            metareview = metareview.strip()

        # Iterate and extract 3 reviews only
        i = 0
        extracted_reviews = []
        while i != 3:
            # main_review for 2022
            # 2023 is a set ['summary_of_the_paper', 'strength_and_weaknesses', 'clarity,_quality,_novelty_and_reproducibility', 'summary_of_the_review']
            if 'strength_and_weaknesses' in submission.details['directReplies'][i]['content']:
                x = submission.details['directReplies'][i]['content']['strength_and_weaknesses'] + '-----' + submission.details['directReplies'][i]['content'][
                    'clarity,_quality,_novelty_and_reproducibility'] + '-----' + submission.details['directReplies'][i]['content']['summary_of_the_review']
                text = x
                text = text.replace('\n', '-----')
                text = text.replace('"', '""')
                text = text.strip()
                extracted_reviews.append(text)
            i += 1

        # Write 3 rows to the CSV
        # each row is: id, review_{i}, metareview
        for review in extracted_reviews:
            writer.writerow([id, str(review), str(metareview)])

        # Empty the list
        extracted_reviews.clear()

print("Done!")

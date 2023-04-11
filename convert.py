from unidecode import unidecode


def convert_csv_file(file_path, output_path):
    import pandas as pd
    df = pd.read_csv(file_path, sep=',', engine='python', usecols=['filename', 'words'],
                     keep_default_na=False)

    df['words'] = df['words'].apply(lambda x: unidecode(x))
    df.to_csv(output_path, index=False, header=False, sep=',', )


if __name__ == '__main__':
    convert_csv_file('trainer/data/test/labels.csv', 'trainer/data/test/labels_new.csv')
    convert_csv_file('trainer/data/train/labels.csv', 'trainer/data/train/labels_new.csv')

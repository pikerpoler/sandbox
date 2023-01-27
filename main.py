import os


def display_bmp(file_path):
    from PIL import Image
    image = Image.open(file_path)
    image.show()


def image_folder_to_vid(folder_path, vid_path, xlsx_path=None):
    from PIL import Image
    import cv2
    import numpy as np
    import glob
    import os
    if xlsx_path is not None:
        import pandas as pd
        df = pd.read_excel(xlsx_path)

    def extract_labels(df, image_path):

        # print(image_path)
        tmp_df = df[df['image path'] == image_path][['label', 'x1', 'y1', 'x2', 'y2']]
        return tmp_df['label'].values, tmp_df[['x1', 'y1']].values.astype(int)[0] , tmp_df[['x2', 'y2']].values.astype(int)[0]

    img_array = []
    for filename in sorted(glob.glob(folder_path + '/*.bmp'), key=lambda x: int(os.path.basename(x).split('.')[1])):
        img = cv2.imread(filename)
        height, width, layers = img.shape
        size = (width, height)

        if xlsx_path is not None:
            folder_name = os.path.basename(folder_path)
            print(os.path.basename(filename))
            image_path = 'data\cat1data\\' + folder_name + '\\' + os.path.basename(filename)
            label, upper_left, lower_right = extract_labels(df, image_path)
            # choose color based on label. 0 is red, 1 is green, 2 is blue, 3 is yellow, 4 is grey
            colors = [(0, 0, 255), (0, 255, 0), (255, 0, 0), (0, 255, 255), (128, 128, 128)]
            color = colors[label[0]]
            # draw a bounding box on the image
            cv2.rectangle(img, tuple(upper_left), tuple(lower_right), color, 2)


        for i in range(10):
            img_array.append(img)

    out = cv2.VideoWriter(vid_path, cv2.VideoWriter_fourcc(*'DIVX'), 15, size)

    for i in range(len(img_array)):
        out.write(img_array[i])
    out.release()


if __name__ == '__main__':

    data_path = '/Users/nadav.nissim/Desktop/cat1data/enc_1.3.12.2.1107.5.4.5.135313.30000020110905191253100000070.512'
    image_folder_to_vid(data_path, 'vid.avi', '/Users/nadav.nissim/Desktop/cat1data/cat1.xlsx')

    # for file in os.listdir(data_path):
    #     if file.endswith(".bmp"):
    #         display_bmp(os.path.join(data_path, file))
    # display_bmp('/Users/nadav.nissim/Desktop/cat1data/enc_1.3.12.2.1107.5.4.5.135313.30000020110905191253100000070.512/img.0.bmp')


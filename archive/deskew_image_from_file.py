def deskew_img_from_file(pageName, img, path_skewfiles):
    page_skewFiles = glob(f'{pageName}*', root_dir=path_skewfiles)
    angle_mode = 0
    if page_skewFiles:
        angles = []
        for skewFile in page_skewFiles:
            with open(skewFile, 'r') as f:
                angles.append(float(f.readline().strip('\n')))
        
        angles, counts = np.unique(angles, return_counts=True)
        angle_mode = angles[np.argmax(counts)]
        img = img.rotate(angle_mode, expand=True, fillcolor='white', resample=Image.Resampling.BICUBIC)

    return img, angle_mode
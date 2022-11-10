'''
An auxiliary script to generate HTML files for image visualization.
'''

import argparse
import os
import shutil

import dominate
from dominate.tags import h3, img, table, td, tr


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_dirs', type=str, nargs='+', required=True)
    parser.add_argument('--output_root', type=str, required=True)
    parser.add_argument('--aliases', type=str, default=None, nargs='+')
    parser.add_argument('--title', type=str, default=None)
    parser.add_argument('--hard_copy', action='store_true')
    args = parser.parse_args()
    if args.aliases is not None:
        assert len(args.image_dirs) == len(args.aliases)
    else:
        args.aliases = [str(i) for i in range(len(args.image_dirs))]
    return args


def check_existence(image_dirs, filename):
    for image_dir in image_dirs:
        if not os.path.exists(os.path.join(image_dir, filename)):
            print(os.path.join(image_dir, filename))
            return False
    return True


if __name__ == "__main__":
    args = get_args()
    filenames = sorted(os.listdir(args.image_dirs[0]))
    filenames = [filename for filename in filenames if filename.endswith('.png')]
    doc = dominate.document(title='Visualization' if args.title is None else args.title)
    if args.title:
        with doc:
            h3(args.title)
    t_main = table(border=1, style="table-layout: fixed;")
    for i, filename in enumerate(filenames):
        bname = filename.replace('.png', '')
        if not check_existence(args.image_dirs, filename):
            continue
        title_row = tr()
        _tr = tr()
        for image_dir, alias in zip(args.image_dirs, args.aliases):
            title_row.add(td('%s-%s' % (alias, bname)))
            _td = td(style="word-wrap: break-word;", halign="center", valign="top")
            source_path = os.path.abspath(os.path.join(image_dir, filename))
            target_path = os.path.abspath(os.path.join(os.path.join(args.output_root, 'images', alias, filename)))
            os.makedirs(os.path.dirname(os.path.abspath(target_path)), exist_ok=True)
            if args.hard_copy:
                shutil.copy(source_path, target_path)
            else:
                os.symlink(source_path, target_path)
            _td.add(img(style="width:256px", src=os.path.relpath(target_path, args.output_root)))
            _tr.add(_td)
        t_main.add(title_row)
        t_main.add(_tr)
    with open(os.path.join(args.output_root, 'viz.html'), 'w') as f:
        f.write(t_main.render())

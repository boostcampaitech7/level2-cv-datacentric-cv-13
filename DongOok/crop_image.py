{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "vscode": {
     "languageId": "plaintext"
    }
   },
   "outputs": [],
   "source": [
    "def get_crop_coord_use_gt(self, image, polygons):\n",
    "    min_x, min_y = min(p[0][0][0] for p in polygons), min(p[0][0][1] for p in polygons)\n",
    "    max_x, max_y = max(p[0][2][0] for p in polygons), max(p[0][2][1] for p in polygons)\n",
    "    h, w, c = image.shape\n",
    "    crop_w, crop_h = max_x - min_x, max_y - min_y\n",
    "    new_min_x = 0 if min_x - ((w - crop_w) / 10) < 0 else min_x - ((w - crop_w) / 10)\n",
    "    new_max_x = w if max_x + ((w - crop_w) / 10) > w else max_x + ((w - crop_w) / 10)\n",
    "    new_min_y = 0 if min_y - ((h - crop_h) / 4) < 0 else min_y - ((h - crop_h) / 4)\n",
    "    new_max_y = h if max_y + ((h - crop_h) / 4) > h else max_y + ((h - crop_h) / 4)\n",
    "    new_min_x, new_min_y, new_max_x, new_max_y = int(new_min_x), int(new_min_y), int(new_max_x), int(new_max_y)\n",
    "    return [new_min_x, new_min_y, new_max_x, new_max_y]"
   ]
  }
 ],
 "metadata": {
  "language_info": {
   "name": "python"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 2
}

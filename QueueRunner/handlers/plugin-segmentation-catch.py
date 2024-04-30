from .CATCH.catch_inference import inference as inference_CATCH
import logging
from typing import Callable
import time
import os
import zipfile
import numpy as np
from tqdm import tqdm
import h5py
from pathlib import Path
import cv2
import pyvips
from exact_sync.v1.models import PluginResultBitmap, PluginResult, PluginResultEntry, Plugin, PluginJob, Image

update_steps = 10 # after how many steps will we update the progress bar during upload (stage1 and stage2 updates are configured in the respective files)

def inference(apis:dict, job:PluginJob, update_progress:Callable, **kwargs):

        image = apis['images'].retrieve_image(job.image)
        logging.info('Retrieving image set for job %d ' % job.id)


        update_progress(0.01)
        
        unlinklist=[] # files to delete

        imageset = image.image_set

 
        logging.info('Checking annotation type availability for job %d' % job.id)
        annotationtypes = {anno_type['name']:anno_type for anno_type in apis['manager'].retrieve_annotationtypes(imageset)}
        
                    
        # The correct annotation type is required in order to be able to add the annotation
        # CAVE: The annotation type also needs to be a part of the product that you want to apply
        # the detection on.
        annoclasses={}
        match_dict = ['TUMOR', 'EPIDERMIS', 'DERMIS', 'SUBCUTIS', 'INFLAMM/NECROSIS']
        for t in annotationtypes:
            for label_class in match_dict:
                if label_class in t.upper():
                    annoclasses[label_class] = annotationtypes[t]
        
        missing = list(set(match_dict) - set(annoclasses.keys()))
        if len(missing) > 0:
            warning = 'Warning: Missing annotation type(s)'
            error_detail = 'Annotation class {} does not exist for imageset '.format(' and '.join(missing))+str(imageset)
            logging.warning(str(error_detail))

        try:
            tpath = os.path.join(os.getcwd(), 'QueueRunner', 'tmp', image.filename)

            if not os.path.exists(tpath):
                if ('.mrxs' in str(image.filename).lower()):
                    tpath = tpath + '.zip'
                logging.info('Downloading image %s to %s' % (image.filename,tpath))
                apis['images'].download_image(job.image, target_path=tpath, original_image=False)
                if ('.mrxs' in str(image.filename).lower()):
                    logging.info('Unzipping MRXS image %s' % (tpath))

                    with zipfile.ZipFile(tpath, 'r') as zip_ref:
                        zip_ref.extractall('tmp/')
                        for f in zip_ref.filelist:
                            unlinklist.append('tmp/'+f.orig_filename)
                        unlinklist.append(tpath)
                    # Original target path is MRXS file
                    tpath = os.path.join(os.getcwd(), 'QueueRunner', 'tmp', image.filename)
                        
        except Exception as e:
            error_message = 'Error: '+str(type(e))+' while downloading'
            error_detail = str(e)
            logging.error(str(e))
            apis['processing'].partial_update_plugin_job(id=job.id, error_message=error_message, error_detail=error_detail)
            return False            

        try:
            logging.info('Inference for job %d' % job.id)
            hdf5_path = os.path.join(os.getcwd(), 'QueueRunner', 'tmp', "{}.hdf5".format(Path(image.filename).stem))
            hdf5_file = h5py.File(hdf5_path, "w")
            inference_CATCH(tpath, hdf5_file, update_progress=update_progress)
            if hdf5_file.__bool__():
                hdf5_file.close()
        except Exception as e:
            error_message = 'Error: '+str(type(e))+' while processing segmentation inference'
            error_detail = str(e)
            logging.error(str(e))
            apis['processing'].partial_update_plugin_job(id=job.id, error_message=error_message, error_detail=error_detail)
            return False

            
        #try:
        #    logging.info('Uploading segmentation prediction %d ' % job.id)
        #    uploaded_image = apis['images'].create_image(file_path=hdf5_path, image_set=imageset)

        #except Exception as e:
        #    error_message = 'Error: '+str(type(e))+' while uploading hdf5 file.'
        #    error_detail = str(e)
        #    logging.error(str(e))
            
        #    apis['processing'].partial_update_plugin_job(id=job.id, error_message=error_message, error_detail=error_detail)
        #    return False

        #"""
        try:
            logging.info('Creating plugin result')
            existing = [j.id for j in apis['processing'].list_plugin_results().results if j.job==job.id]
            if len(existing)>0:
                apis['processing'].destroy_plugin_result(existing[0])
            
            # Create Result for job
            # Each job is linked to a single result, which may consist of several result entries.
            result = PluginResult(job=job.id, image=image.id, plugin=job.plugin, entries=[])
            result = apis['processing'].create_plugin_result(body=result)

            
            logging.info('Creating plugin entry')
        except Exception as e:
            error_message = 'Error: '+str(type(e))+' while creating plugin result'
            error_detail = str(e)+f'Job {job.id}, Image {image.id}, Pliugin {job.plugin}'
            logging.error(str(e))
            
            apis['processing'].partial_update_plugin_job(id=job.id, error_message=error_message, error_detail=error_detail)
            return False
            
        try:

            # Create result entry for result
            # Each plugin result can contain collection of annotations. 
            resultentry = PluginResultEntry(pluginresult=result.id, name='Segmentation', annotation_results = [], bitmap_results=[], default_threshold=0.0) # optionally set threshold
            resultentry = apis['processing'].create_plugin_result_entry(body=resultentry)

        except Exception as e:
            error_message = 'Error: '+str(type(e))+' while creating plugin result entry'
            error_detail = str(e)+f'PluginResult {result.id}'
            logging.error(str(e))
            
            apis['processing'].partial_update_plugin_job(id=job.id, error_message=error_message, error_detail=error_detail)
            return False

        
        try:
            with h5py.File(str(hdf5_path), 'r') as hf:
                for n, key in enumerate(list(hf.keys())):
                    data = hf[key]
                    ndarray_data = np.array(data)
                    scaled_image_data = (ndarray_data * (255 / len(np.unique(ndarray_data)))).astype(np.uint8)


                    # Define a color mapping for each integer value
                    color_map = {
                        0: (0, 0, 0, 255),    # Black
                        1: (255, 0, 0, 255),  # Red
                        2: (0, 255, 0, 255),  # Green
                        3: (0, 0, 255, 255),  # Blue
                        4: (255, 255, 0, 255),  # Yellow
                        5: (255, 255, 255, 255)  # White
                    }

                    #colored_image = cv2.applyColorMap(ndarray_data.astype(np.uint8), colormap=color_map)
                    colored_image = cv2.applyColorMap(scaled_image_data, cv2.COLORMAP_VIRIDIS)
                    vi = pyvips.Image.new_from_array(colored_image)
                    mask_path = os.path.join(os.getcwd(), 'QueueRunner', 'tmp', "{}_{}.tiff".format(Path(image.filename).stem, key))
                    vi.tiffsave(str(mask_path), tile=True, compression='lzw', bigtiff=True, pyramid=True, tile_width=256, tile_height=256)
                    image_type = int(Image.ImageSourceTypes.DEFAULT)
                    #image = apis['images'].create_image(file_path=mask_path, image_type=image_type, image_set=imageset).results[0]
                    apis['images'].create_image(file_path='/Volumes/SLIDES/Slides/SegmentationMultiScanner/CS2/SCC/scc_01_cs2.svs', image_type=image_type, image_set=imageset, async_req=True)
                    print()

                    """


                    mask = Image.new("RGB", (ndarray_data.shape[1], ndarray_data.shape[0]))

                    # Set pixel colors based on the values in the array
                    for y in range(ndarray_data.shape[0]//100):
                        for x in range(ndarray_data.shape[1]//100):
                            pixel_value = data[y][x]
                            color = color_map[pixel_value]
                            mask.putpixel((x, y), color)
                    
                    mask_path = os.path.join(os.getcwd(), 'QueueRunner', 'tmp', "{}_{}.bmp".format(Path(image.filename).stem, key))
                    mask.save(mask_path)

                    #scaled_image_data = (ndarray_data/np.max(ndarray_data))
                    #scaled_image_data = (ndarray_data * (255 / len(np.unique(ndarray_data)))).astype(np.uint8)
                    #colored_image = cv2.applyColorMap(scaled_image_data, cv2.COLORMAP_VIRIDIS)
                    #colored_image = cv2.cvtColor(colored_image, cv2.COLOR_BGR2RGB)
                    #cv2.imwrite(mask_path, colored_image)
                    #cv2.imwrite(mask_path, scaled_image_data)
                    bitmap = PluginResultBitmap(bitmap=mask_path, channels=4, default_alpha= 0.5, default_threshold=0.0, name='CATCH {}'.format(key),  pluginresultentry=resultentry.id, image=image.id)
                    bitmap = apis['processing'].create_plugin_result_bitmap(body=bitmap, async_req=True)
                    update_progress (90+10*(n/len(list(hf.keys()))))
                    """            
        except Exception as e:
            error_message = 'Error: '+str(type(e))+' while uploading the annotations'
            error_detail = str(e)
            logging.error(str(e))
            
            apis['processing'].partial_update_plugin_job(id=job.id, error_message=error_message, error_detail=error_detail)
            return False
        
        try:
            os.unlink(tpath)
            os.unlink(hdf5_path)
            #os.unlink(mask_path)
            for f in unlinklist:
                os.unlink(f)
        
        except Exception as e:
            logging.error('Error while deleting files: '+str(e)+'. Continuing anyway.')
        
        return True


plugin = {  'name':'CATCH segmentation baseline',
            'author':'Frauke Wilm', 
            'package':'science.imig.catch', 
            'contact':'frauke.wilm@fau.de', 
            'abouturl':'https://github.com/DeepPathology/MIDOG_reference_docker', 
            'icon':'QueueRunner/handlers/CATCH/catch_logo.jpg',
            'products':[],
            'results':[],
            'inference_func' : inference}



from setuptools import setup

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name='umt',
    version='0.0.1',
    author='Nathan A. Rooy',
    author_email='nathanrooy@gmail.com',
    url='https://github.com/nathanrooy/rpi-urban-mobility-tracker',
    description='The easiest way to count pedestrians, cyclists, and vehicles on edge computing devices.',
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=['umt'],
    python_requires='>=3.7',
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent'
    ],
    install_requires=[
		'filterpy',
		'imutils',
		'numpy',
		'pillow',
	    	'opencv-python',
                'prometheus_client',
		'scipy',
		'scikit-image',
	    	'tensorflow'
    ],
    entry_points={
        'console_scripts': [
            'umt = umt.umt_main:main'
        ]
    },
    package_data={
    	'umt':[
    		'models/tflite/coco_ssd_mobilenet_v1_1.0_quant_2018_06_29/labelmap.txt',
    		'models/tflite/coco_ssd_mobilenet_v1_1.0_quant_2018_06_29/detect.tflite',
    		'models/tpu/mobilenet_ssd_v2_coco_quant/coco_labels.txt',
    		'models/tpu/mobilenet_ssd_v2_coco_quant/mobilenet_ssd_v2_coco_quant_postprocess_edgetpu.tflite',
		'deep_sort/*'
    		]
    },
)

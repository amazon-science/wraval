#
# // Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# // SPDX-License-Identifier: Apache-2.0
#
import os
import logging
import warnings

# Suppress Pydantic warning
warnings.filterwarnings("ignore", message="Field name \"json\" in \"MonitoringDatasetFormat\" shadows an attribute in parent \"Base\"")

# Configure logging before any AWS imports
logging.getLogger('sagemaker').setLevel(logging.ERROR)
logging.getLogger('sagemaker.config').setLevel(logging.ERROR)  # Specifically target the config module
logging.getLogger('boto3').setLevel(logging.ERROR)
logging.getLogger('botocore').setLevel(logging.ERROR)
logging.getLogger('urllib3').setLevel(logging.ERROR)

# Suppress AWS credential messages
os.environ['SAGEMAKER_SUPPRESS_DEFAULTS'] = 'true'
os.environ['AWS_SDK_LOAD_CONFIG'] = '0'  # Suppress AWS SDK config loading messages

# Now import AWS modules
import boto3
import sagemaker 
#
# // Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# // SPDX-License-Identifier: Apache-2.0
#
import boto3

def get_current_aws_account_id():
    sts = boto3.client('sts')
    identity = sts.get_caller_identity()
    return identity['Account']

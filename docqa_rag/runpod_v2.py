"""Opt-in v2 Pod transport, preserving Session's owned-only lifecycle protocol."""
from .lifecycle import RunPodAPI

class RunPodV2API(RunPodAPI):
    base_url='https://api.runpod.io/v2'

    @staticmethod
    def create_payload(p):
        ids=p['gpuTypeIds']
        if len(ids)!=1:raise ValueError('v2 requires one explicit GPU type per create')
        result={'name':p['name'],'image':p['imageName'],'disk':p['containerDiskInGb'],
            'cloud':p['cloudType'],'gpu':{'id':ids[0],'count':p['gpuCount'],
                'allowedCudaVersions':p['allowedCudaVersions'],'minRamPerGpu':8,'minVcpuCountPerGpu':2},
            'entrypoint':p['dockerEntrypoint'],'cmd':p['dockerStartCmd'],'env':p['env'],'ports':p['ports']}
        if p.get('volumeInGb') or p.get('networkVolumeId'):raise ValueError('This session transport allows no persistent volume')
        if p.get('dataCenterIds'):result['dataCenterIds']=p['dataCenterIds']
        return result

    @staticmethod
    def normalize(p):
        if p is None or p.get('status')=='TERMINATED':return None
        return {**p,'costPerHr':p.get('cost'),
            'machine':{'gpuTypeId':p.get('gpu',{}).get('id'),'dataCenterId':p.get('dataCenterId'),'cudaVersion':p.get('cudaVersion')}}

    def request(self,method,path,payload=None):
        if not (path=='/pods' or path.startswith('/pods/')):
            return RunPodAPI(self.key).request(method,path,payload)
        if method=='POST' and path=='/pods':payload=self.create_payload(payload)
        value=super().request(method,path,payload)
        if method=='GET' and path=='/pods':return [self.normalize(p) for p in (value or {}).get('pods',[]) if p.get('status')!='TERMINATED']
        if method in {'GET','POST'}:return self.normalize(value)
        return value

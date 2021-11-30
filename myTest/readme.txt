��װ��
1����װgo����ǰʹ���� go1.8.1����װ�����Ҫ����·������������
2��protobuf: libprotoc 2.4.1
3��golang ����������
��/etc/profile �������ӣ�
export GOPATH=/data/go
export GOPATH=$GOPATH:/data/go_project
export PATH=$PATH:/data/go/bin
export PATH=$PATH:/data/go_project/bin

��Ч��source /etc/profile


Ŀ¼��
1������/data/go_project/
2����1��Ŀ¼�����bin��pkg��src Ŀ¼
3���� /data/go_project/src/uframework ����uframework
2��message copy �� uframework message Ŀ¼��
3��ҵ��Ŀ¼Ҳ��uframework ͬ����ֹ������ustorage
4����ҵ��Ŀ¼�£���ҵ����ģ�飬����chxd��fileidx ��

/data/go_project
           |---bin
           |---pkg
           |---src
                |---uframework
                       |---message
                            |---ucloud
                            |---ustorage
                |---ustorage
				                |---storeidx
				                       |---fileidx
				                       |---store-access
				                |---storebase
				                       |---chx
				                       |---chx-master
				                       |---httpsvr
				                |---ufile
				                       |---ufile-access
				        |--- other-project


���ϲ���Ŀ¼��
1����Ŀ¼��/root/ufilev2/, ��ģ���ƽ����
2��      /root/ufilev2/
                       |---fileidx
                       |---store-access
                       |---chx
                       |---chx-master
                       |---httpsvr
                       |---store-access
                            |---access
                            |---config.json
                            |---log
                            |---tools



SRCDIRS_EXCLUDE=example utils
SRCDIRS_ALL = $(sort $(subst /,,$(dir $(wildcard */))))
SRCDIRS = $(filter-out $(SRCDIRS_EXCLUDE), $(SRCDIRS_ALL))

PKGDIRS_EXCLUDE = $(GOROOT)/pkg
PKGDIRS_ALL = $(addsuffix /pkg, $(subst :, ,$(GOPATH)))
PKGDIRS = $(filter-out $(PKGDIRS_EXCLUDE), $(PKGDIRS_ALL))

PROTO_GEN_GO_PATH = "message/protobuf"
.PHONY:utils

all:protobuf build_common_proto utils 
	@for subdir in $(SRCDIRS);do \
		cd $$subdir; go install; cd $(CURDIR); \
	done 

protobuf:
	@cd $(PROTO_GEN_GO_PATH);make;cd $(CURDIR);

build_common_proto:
	@cd message/proto/ucloud;make;cd $(CURDIR);\
	cd message/proto/uns;make;cd $(CURDIR);

utils:
	@cd utils;make;cd $(CURDIR);

show:
	@echo "==================src====================="
	@echo SRCDIRS_ALL: $(SRCDIRS_ALL)
	@echo SRCDIRS_EXCLUDE: $(SRCDIRS_EXCLUDE)
	@echo SRCDIRS: $(SRCDIRS)
	@echo "==================pkg====================="
	@echo PKGDIRS_EXCLUDE: $(PKGDIRS_EXCLUDE)
	@echo PKGDIRS_ALL: $(PKGDIRS_ALL)
	@echo PKGDIRS: $(PKGDIRS)
	@echo "=================clean===================="
	@for subdir in $(PKGDIRS); do \
		cd $$subdir;\
		result=`find . |grep uframework|head -n1|awk -F"." '{print $$2}'`; \
		if [ -n "$$result" ];then \
			echo CLEAN_PKGDIRS:$$subdir$$result; \
		fi \
	done

clean: 
	@cd message/protobuf;make nuke;cd $(CURDIR);\
	cd message/proto/uns;make clean;cd $(CURDIR); \
	cd message/proto/ucloud;make clean;cd $(CURDIR); \
	cd utils;make clean;cd $(CURDIR);\
	for subdir in $(PKGDIRS); do \
		cd $$subdir;\
		result=`find . |grep uframework|head -n1|awk -F"." '{print $$2}'`; \
		if [ -n "$$result" ];then \
			cd $$subdir$$result; rm -rf *; \
		fi \
	done

include ../common.mk

##### src files
INCEXPORTS  :=  nccl_net.h
LIBSRCFILES :=  net_socket.cc  utils.cc
NCCL_MAJOR = 0
NCCL_MINOR = 1
NCCL_PATCH = 6

##### lib files
LIBNAME     := libtransport.so
LIBTARGET  := $(LIBNAME:%=%.$(NCCL_MAJOR).$(NCCL_MINOR).$(NCCL_PATCH))
LIBSONAME   := $(LIBNAME:%=%.1)
STATICLIBNAME := libtransport_static.a
BUILDDIR ?= $(abspath ../build)
INCDIR := ../include
LIBDIR := $(BUILDDIR)/lib
OBJDIR := $(BUILDDIR)/obj

##### target files
STATICLIBTARGET := $(STATICLIBNAME)
LIBOBJ     := $(LIBSRCFILES:%.cc=$(OBJDIR)/%.o)
DEPFILES   := $(LIBOBJ:%.o=%.d)
LDFLAGS    += -lpthread -lrt -ldl

##### rules
build : lib staticlib

lib : $(INCTARGETS) $(LIBDIR)/$(LIBTARGET) $(PKGDIR)/$(PKGTARGET)

staticlib : $(LIBDIR)/$(STATICLIBTARGET)

# Empty target to force rebuild
ALWAYS_REBUILD:

-include $(DEPFILES)
$(LIBDIR)/$(LIBTARGET) $(LIBDIR)/$(STATICLIBTARGET) : $(LIBOBJ)

$(LIBDIR)/$(LIBTARGET): $(LIBOBJ) 
	@printf "Linking    %-35s > %s\n" $(LIBTARGET) $@
	mkdir -p $(LIBDIR)
	echo $(LIBSONAME)
	$(CXX) $(CXXFLAGS) -shared -Wl,--no-as-needed -Wl,-soname,$(LIBSONAME) -o $@ $(LIBOBJ) $(LDFLAGS)
	ln -sf $(LIBSONAME) $(LIBDIR)/$(LIBNAME)
	ln -sf $(LIBTARGET) $(LIBDIR)/$(LIBSONAME)

$(LIBDIR)/$(STATICLIBTARGET): $(LIBOBJ) 
	@printf "Archiving  %-35s > %s\n" $(STATICLIBTARGET) $@
	mkdir -p $(LIBDIR)
	$(eval TMP := $(shell mktemp -d))
	cp $(LIBOBJ) $(TMP)
	ar cr $@ $(LIBOBJ) $(TMP)/*.o
	rm -Rf $(TMP)

$(OBJDIR)/%.o : %.cc
	@printf "Compiling  %-35s > %s\n" $< $@
	mkdir -p `dirname $@`
	$(CXX) -I. -I$(INCDIR) $(CXXFLAGS) -Iinclude -c $< -o $@
	@$(CXX) -I. -I$(INCDIR) $(CXXFLAGS) -Iinclude -M $< > $(@:%.o=%.d.tmp)
	@sed "0,/^.*:/s//$(subst /,\/,$@):/" $(@:%.o=%.d.tmp) > $(@:%.o=%.d)
	@sed -e 's/.*://' -e 's/\\$$//' < $(@:%.o=%.d.tmp) | fmt -1 | \
                sed -e 's/^ *//' -e 's/$$/:/' >> $(@:%.o=%.d)
	@rm -f $(@:%.o=%.d.tmp)

clean :
	rm -rf ${LIBDIR} ${OBJDIR}

install : lib
	mkdir -p $(PREFIX)/lib
	mkdir -p $(PREFIX)/lib/pkgconfig
	mkdir -p $(PREFIX)/include
	cp -P -v $(BUILDDIR)/lib/lib* $(PREFIX)/lib/
	cp -P -v $(BUILDDIR)/lib/pkgconfig/* $(PREFIX)/lib/pkgconfig/
	cp -v $(BUILDDIR)/include/* $(PREFIX)/include/

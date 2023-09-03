# Python libraries
SOURCE_OBJ = $(wildcard ./obj/*.o)
SOURCE_INC = $(addprefix -I,$(wildcard */inc))
PYTHON_INC_FLAGS = -IC:\ProgramData\anaconda3\include \
-IC:\ProgramData\anaconda3\Library\include
PYTHON_LD_FLAGS = -L C:\ProgramData\anaconda3\libs


# Compiler and compiler flags
CXX := g++
NVCC := nvcc
CXXFLAGS := -std=c++11 -Wall
NVCCFLAGS := -arch=sm_86  # Adjust CUDA architecture as needed
PYLIB := -IC:\ProgramData\anaconda3\include

# Directories
SRCDIRS := beam beamline simulation pywrapper util
INCLUDEDIR := inc
BUILDDIR := obj

INCDIRS := $(foreach dir,$(SRCDIRS),$(dir)/inc)

# Source files and object files
SRCS := $(foreach dir,$(SRCDIRS),$(wildcard $(dir)/src/*.cpp) $(wildcard $(dir)/src/*.cu))
OBJS := $(patsubst %.cpp, $(BUILDDIR)/%.o, $(notdir $(filter %.cpp, $(SRCS))))
CUOBJS := $(patsubst %.cu, $(BUILDDIR)/%.cu.o, $(notdir $(filter %.cu, $(SRCS))))

BEAM_SRCDIR = beam/src
BEAMLINE_SRCDIR = beamline/src
SIMULATION_SRCDIR = simulation/src
PYWRAPPER_SRCDIR = pywrapper/src
UTIL_SRCDIR = util/src

# Source and object files
BEAM_SOURCES = $(wildcard $(BEAM_SRCDIR)/*.cpp $(BEAM_SRCDIR)/*.cu)
BEAM_OBJS = $(patsubst %.cpp, $(BUILDDIR)/%.o, $(notdir $(filter %.cpp, $(BEAM_SOURCES))))
BEAM_CUOBJS = $(patsubst %.cu, $(BUILDDIR)/%.cu.o, $(notdir $(filter %.cu, $(BEAM_SOURCES))))

BEAMLINE_SOURCES = $(wildcard $(BEAMLINE_SRCDIR)/*.cpp $(BEAMLINE_SRCDIR)/*.cu)
BEAMLINE_OBJS = $(patsubst %.cpp, $(BUILDDIR)/%.o, $(notdir $(filter %.cpp, $(BEAMLINE_SOURCES))))
BEAMLINE_CUOBJS = $(patsubst %.cu, $(BUILDDIR)/%.cu.o, $(notdir $(filter %.cu, $(BEAMLINE_SOURCES))))

SIMULATION_SOURCES = $(wildcard $(SIMULATION_SRCDIR)/*.cpp $(SIMULATION_SRCDIR)/*.cu)
SIMULATION_OBJS = $(patsubst %.cpp, $(BUILDDIR)/%.o, $(notdir $(filter %.cpp, $(SIMULATION_SOURCES))))
SIMULATION_CUOBJS = $(patsubst %.cu, $(BUILDDIR)/%.cu.o, $(notdir $(filter %.cu, $(SIMULATION_SOURCES))))

PYWRAPPER_SOURCES = $(wildcard $(PYWRAPPER_SRCDIR)/*.cpp $(PYWRAPPER_SRCDIR)/*.cu)
PYWRAPPER_OBJS = $(patsubst %.cpp, $(BUILDDIR)/%.o, $(notdir $(filter %.cpp, $(PYWRAPPER_SOURCES))))
PYWRAPPER_CUOBJS = $(patsubst %.cu, $(BUILDDIR)/%.cu.o, $(notdir $(filter %.cu, $(PYWRAPPER_SOURCES))))

UTIL_SOURCES = $(wildcard $(UTIL_SRCDIR)/*.cpp $(UTIL_SRCDIR)/*.cu)
UTIL_OBJS = $(patsubst %.cpp, $(BUILDDIR)/%.o, $(notdir $(filter %.cpp, $(UTIL_SOURCES))))
UTIL_CUOBJS = $(patsubst %.cu, $(BUILDDIR)/%.cu.o, $(notdir $(filter %.cu, $(UTIL_SOURCES))))


# Output DLL
TARGET := HpsimLib.dll

# Build rules
all: $(TARGET)

$(TARGET):HpsimLib.cpp $(BEAM_OBJS) $(BEAM_CUOBJS) $(BEAMLINE_OBJS) $(BEAMLINE_CUOBJS) $(SIMULATION_OBJS) $(SIMULATION_CUOBJS) $(PYWRAPPER_OBJS) $(PYWRAPPER_CUOBJS) $(UTIL_OBJS) $(UTIL_CUOBJS) 
	nvcc $(NVCCFLAGS) -c HpsimLib.cpp -o HpsimLib.obj $(PYTHON_INC_FLAGS) $(SOURCE_INC)
	nvcc $(NVCCFLAGS) -shared $(SOURCE_INC) $(PYTHON_INC_FLAGS) $(PYTHON_LD_FLAGS) $(SOURCE_OBJ)  HpsimLib.obj -o $@

# 检查并创建目标文件夹
$(shell IF NOT EXIST $(BUILDDIR) mkdir $(BUILDDIR))

$(BUILDDIR)/%.o: $(BEAM_SRCDIR)/%.cpp
	$(NVCC) $(NVCCFLAGS) $(addprefix -I, $(INCDIRS)) $(PYLIB) -c -o $@ $<

$(BUILDDIR)/%.cu.o: $(BEAMLINE_SRCDIR)/%.cu
	$(NVCC) $(NVCCFLAGS) $(addprefix -I, $(INCDIRS)) $(PYLIB) -c -o $@ $<


$(BUILDDIR)/%.o: $(BEAMLINE_SRCDIR)/%.cpp
	$(NVCC) $(NVCCFLAGS) $(addprefix -I, $(INCDIRS)) $(PYLIB) -c -o $@ $<

$(BUILDDIR)/%.cu.o: $(BEAM_SRCDIR)/%.cu
	$(NVCC) $(NVCCFLAGS) $(addprefix -I, $(INCDIRS)) $(PYLIB) -c -o $@ $<


$(BUILDDIR)/%.o: $(SIMULATION_SRCDIR)/%.cpp
	$(NVCC) $(NVCCFLAGS) $(addprefix -I, $(INCDIRS)) $(PYLIB) -c -o $@ $<

$(BUILDDIR)/%.cu.o: $(SIMULATION_SRCDIR)/%.cu
	$(NVCC) $(NVCCFLAGS) $(addprefix -I, $(INCDIRS)) $(PYLIB) -c -o $@ $<


$(BUILDDIR)/%.o: $(PYWRAPPER_SRCDIR)/%.cpp
	$(NVCC) $(NVCCFLAGS) $(addprefix -I, $(INCDIRS)) $(PYLIB) -c -o $@ $<

$(BUILDDIR)/%.cu.o: $(PYWRAPPER_SRCDIR)/%.cu
	$(NVCC) $(NVCCFLAGS) $(addprefix -I, $(INCDIRS)) $(PYLIB) -c -o $@ $<


$(BUILDDIR)/%.o: $(UTIL_SRCDIR)/%.cpp
	$(NVCC) $(NVCCFLAGS) $(addprefix -I, $(INCDIRS)) $(PYLIB) -c -o $@ $<

$(BUILDDIR)/%.cu.o: $(UTIL_SRCDIR)/%.cu
	$(NVCC) $(NVCCFLAGS) $(addprefix -I, $(INCDIRS)) $(PYLIB) -c -o $@ $<


# Clean rule
clean:
	rm -rf $(BUILDDIR) $(TARGET)

.PHONY: all clean

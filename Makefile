SOURCE = $(wildcard obj/*.obj)
# Your python folders
PYTHON_INC_FLAGS = -ID:\ProgramFiles\anaconda2022\include \
-ID:\ProgramFiles\anaconda2022\Library\include -I.\inc
PYTHON_LD_FLAGS = -L D:\ProgramFiles\anaconda2022\libs

HpsimLib.dll:HpsimLib.cpp
	nvcc -m64 -c $< -o HpsimLib.obj $(PYTHON_INC_FLAGS)
	nvcc -m64 -shared $(PYTHON_INC_FLAGS) $(PYTHON_LD_FLAGS) $(SOURCE_BEAM) $(SOURCE2) HpsimLib.obj -o $@

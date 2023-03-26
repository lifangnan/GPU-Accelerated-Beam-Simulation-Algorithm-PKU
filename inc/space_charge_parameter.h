#ifndef SPACE_CHARGE_PARAMETER_H
#define SPACE_CHARGE_PARAMETER_H

struct SpaceChargeParam
{
  unsigned int nx, ny, nz;          // scheff sizes
};

struct ScheffMeshSize
{
  double dr;
  double dz;
};
#endif

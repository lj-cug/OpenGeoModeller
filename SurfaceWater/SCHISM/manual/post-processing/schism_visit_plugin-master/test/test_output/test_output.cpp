// test_scribio2.cpp : Defines the entry point for the console application.
//

#include "SCHISMFile10.h"
#include "MeshConstants10.h"
#include "SCHISMFileUtil10.h"
#include "SchismGeometry10.h"
#include "SCHISMMeshProvider10.h"
#include "ZCoordFileMeshProvider10.h"
#include "NetcdfSCHISMOutput10.h"
#include <iostream>
#include <fstream>
#include <cmath>
using namespace std;

int test_netcdf4(std::string &a_ncfile)
{

	NcFile *ncfile_ptr = new NcFile(a_ncfile.c_str(), NcFile::ReadOnly);
	if (ncfile_ptr->is_valid() == 0)
	{
		cout << "Error:" << a_ncfile << " is not opened correctly" << endl;
		return 0;
	}
	else
	{
		cout << a_ncfile << " is opened correctly" << endl;
		return 1;
	}
}

// those commented lines are not compatible with current
// code NC C++ API
// this function testing open nc file using NC C++ 4.3 api
// which changes greatly from 4.1. This function should also
// catch Long int attribute error which not support well by 4.1
//  to be compiled, project should link to NC C++ 4.3.1 lib
// instead of older version.
// void test_nc43_api(std::string outfile)
//{
//	//NcFile ncFile(soutoutFile,NcFile::read);
//	//fprintf(stderr,"***Fail: %s\n",nc_strerror(2));
//	NcFile ncFile(outfile,NcFile::read);
//	multimap<string,NcVar> all_vars;
//	multimap<string,NcVar>::iterator iter;
//	all_vars=ncFile.getVars();
//	int num_var=ncFile.getVarCount();
//
//	for(iter=all_vars.begin();iter!=all_vars.end();iter++)
//	{
//		NcVar  a_var=(*iter).second;
//		map<string,NcVarAtt> all_atts;
//	    map<string,NcVarAtt>::iterator att_iter;
//
//		all_atts=a_var.getAtts();
//
//		for(att_iter=all_atts.begin();att_iter!=all_atts.end();att_iter++)
//		{
//			NcVarAtt a_att=(*att_iter).second;
//			NcType type=a_att.getType();
//			if (type==ncChar)
//			{
//				std::string aa;
//				a_att.getValues(aa);
//			}
//			else if ((type==ncShort)||(type==ncInt))
//			{
//				int aa;
//				a_att.getValues(&aa);
//			}
//			else if(type==ncInt64)
//			{
//				long aa;
//				a_att.getValues(&aa);
//				cout<<"long int "<<aa<<"\n";
//			}
//			else if(type==ncFloat)
//			{
//				float aa;
//				a_att.getValues(&aa);
//			}
//			else if(type==ncDouble)
//			{
//				double aa;
//				a_att.getValues(&aa);
//			}
//
//		}
//
//	cout<<"NC C++ 4.3 API passed "<<std::endl;
//}
//

int testScribeIO(std::string &soutputFile, std::string &meshFile)
{
	// NetcdfSchismOutput10  * soutPtr = new NetcdfSchismOutput10(soutputFile);
	NetcdfSchismOutput10 *soutPtr = new NetcdfSchismOutput10(meshFile);
	SCHISMMeshProvider10 *meshPtr = new ZCoordMeshProvider10(meshFile);

	int numNode = meshPtr->numberOfNode();
	int numLayer = meshPtr->numberOfLayer();
	int *nodes_bottom = new int[numNode];

	double *stemp = new double[numNode * numLayer];
	soutPtr->set_mesh_bottom(meshPtr->get_mesh_data_ptr(), 0);
	soutPtr->get_node_bottom(nodes_bottom, 0);
	SCHISMVar10 *svarptr = soutPtr->get_var("salinity");
	svarptr->set_cur(0);
	svarptr->get(stemp);

	std::string toutputFile = "E:\\temp\\new_nc2\\temperature_1.nc";
	NetcdfSchismOutput10 *toutPtr = new NetcdfSchismOutput10(toutputFile);
	toutPtr->set_mesh_bottom(meshPtr->get_mesh_data_ptr(), 0);
	toutPtr->get_node_bottom(nodes_bottom, 0);
	SCHISMVar10 *tvarptr = toutPtr->get_var("temperature");
	tvarptr->set_cur(1);
	tvarptr->get(stemp);

	NetcdfSchismOutput10 *out2dPtr = new NetcdfSchismOutput10(meshFile);
	out2dPtr->set_mesh_bottom(meshPtr->get_mesh_data_ptr(), 0);
	out2dPtr->get_node_bottom(nodes_bottom, 0);
	cout << "passed\n";
	delete nodes_bottom;
	delete stemp;
	return 0;
}

int testZCoreProvider(std::string &file1, std::string &out)
{

	ifstream *f1 = new ifstream(file1, ios::binary);
	cout << f1->is_open() << " \n";

	delete f1;

	std::string mdoutputFile = "E:\\temp\\new_nc2\\salinity_1.nc";
	std::string meshFile = "E:\\temp\\new_nc2\\out2d_1.nc";

	SCHISMMeshProvider10 *meshPtr = new ZCoordMeshProvider10(meshFile);

	int numNode = meshPtr->numberOfNode();
	int numCell = meshPtr->numberOfElement();
	int nodePerCell = meshPtr->numberOfNodesPerCell();
	int numLayer = meshPtr->numberOfLayer();

	cout << "num of node:" << meshPtr->numberOfNode() << "\n";
	cout << "num of cell:" << meshPtr->numberOfElement() << "\n";
	cout << "num of node per cell :" << meshPtr->numberOfNodesPerCell() << "\n";
	cout << "num of layer:" << numLayer << "\n";

	ofstream outfile;
	outfile.open(out);

	long *meshElementNode = new long[numCell * (nodePerCell + 1)];

	meshPtr->fillMeshElement(meshElementNode);

	for (int i = 0; i < numCell; i++)
	{
		// outfile<<"ele "<<i;
		int numNodeT = meshElementNode[i * (nodePerCell + 1)];
		for (int j = 0; j < numNodeT; j++)
		{
			// outfile<<" "<<meshElementNode[i*(nodePerCell+1)+1+j];
		}
		// outfile<<"\n";
	}

	long *sideNode = NULL;
	int numberSide = 0;
	numberSide = meshSideNum(meshElementNode, numCell, numNode);
	sideNode = new long[numberSide * 2];
	meshSideNode(sideNode, meshElementNode, numberSide, numCell, numNode);

	for (int i = 0; i < numberSide; i++)
	{
		// outfile<<i<<" "<<sideNode[i*2]<<" "<<sideNode[i*2+1]<<"\n";
	}
	int *kbs = new int[numberSide];
	meshPtr->fillKbs(kbs, 0);
	int *kbp = new int[numNode];
	meshPtr->fillKbp00(kbp, 0);

	float *nodeCoord3D = new float[3 * numLayer * numNode];
	float *nodeCoordPtr = nodeCoord3D;
	meshPtr->fillPointCoord3D(nodeCoord3D, 0);
	int *zvar_size = new int[numNode];
	int *start_id = new int[numNode];
	for (int i = 0; i < numNode; i++)
	{
		zvar_size[i] = 0;
		start_id[i] = 0;
	}
	for (int i = 0; i < numLayer; i++)
	{
		outfile << "Layer:" << i << "\n";
		for (int j = 0; j < numNode; j++)
		{

			{
				if (i < (kbp[j] - 1))
				{
					if (j == 45)
						outfile << "node " << j << " NA \n";
				}
				else
				{
					float x = *nodeCoordPtr++;
					float y = *nodeCoordPtr++;
					float z = *nodeCoordPtr++;
					if (j == 45)
						outfile << "node " << j << " "
								<< " " << x << " " << y << " " << z << "\n";
					zvar_size[j]++;
				}
			}
		}
	}

	// count total size of z core
	int valid3dnode = 0;

	for (int j = 0; j < numNode; j++)
	{
		valid3dnode = valid3dnode + zvar_size[j];
	}

	float *z3d = new float[valid3dnode];

	// test zcoords3d

	meshPtr->zcoords3D(z3d, 0);

	delete z3d;

	outfile << "var z size\n";
	for (int i = 0; i < numNode; i++)
	{
		if (i > 0)
			start_id[i] = start_id[i - 1] + zvar_size[i];
		outfile << i << " " << start_id[i] << " \n";
	}

	// try to get side x,y,z from meshprovider

	float *sideCoord3D = new float[3 * numLayer * numberSide];
	meshPtr->fillSideCenterCoord3D(sideCoord3D, 0);

	float *sideCoordPtr = sideCoord3D;

	for (int i = 0; i < numLayer; i++)
	{
		// outfile<<"Layer:"<<i<<"\n";
		for (int j = 0; j < numberSide; j++)
		{
			if (i < (kbs[j] - 1))
			{
				// outfile<<"side "<<j<<" NA \n";
			}
			else
			{
				float x = *sideCoordPtr++;
				float y = *sideCoordPtr++;
				float z = *sideCoordPtr++;
				// outfile<<"side "<<j<<" "<<sideNode[j*2]<<" "<<sideNode[j*2+1]<<" "<<x<<" "<<y<<" "<<z<<"\n";
			}
		}
	}

	// test maping 3d 2d element
	int *kbe = new int[numCell];
	meshPtr->fillKbe(kbe, 0);
	int *eleto3DEles = new int[numCell * numLayer];
	float *zptr = new float[numCell * numLayer];
	meshPtr->zEleCenter3D(zptr, 0);
	int Index = 0;
	// outfile<<" zptr \n";
	for (int iLayer = 0; iLayer < numLayer; iLayer++)
	{
		for (int iEle = 0; iEle < numCell; iEle++)
		{
			int bottomLayer = std::max(1, kbe[iEle]);
			if (bottomLayer <= (iLayer + 1))
			{
				eleto3DEles[iLayer + iEle * numLayer] = Index;
				// outfile<<zptr[Index]<<" ";
				Index++;
			}
			else
			{
				// outfile<<"NA ";
			}
		}
		// outfile<<"\n";
	}
	outfile << "element z information\n";
	for (int iEle = 0; iEle < numCell; iEle++)
	{
		for (int iLayer = 0; iLayer < numLayer; iLayer++)
		{
			int bottomLayer = std::max(1, kbe[iEle]);
			if (bottomLayer <= (iLayer + 1))
			{
				Index = eleto3DEles[iLayer + iEle * numLayer];
				// outfile<<Index<<" ";
				// outfile<<zptr[Index]<<" ";
			}
			else
			{
				// outfile<<" NA ";
			}
		}
		// outfile<<"\n";
	}

	delete sideNode;
	delete kbs;
	delete kbp;
	delete kbe;
	delete sideCoord3D;
	delete eleto3DEles;
	delete zptr;
	outfile.close();
	delete meshPtr;
	return 0;
}

void main(int argc, char *argv[])
{
	std::string meshFile;
	std::string soutputFile;
	// The first arg is an out2d file, if it exists.
	// The second one is a SCHISM file to test.
	meshFile = argc > 1 ? argv[1] : "E:/temp/test/b1/out2d_1.nc";
	soutputFile = argc > 2 ? argv[2] : "E:/temp/test/depth_averaged_salinity_75.nc";
	test_netcdf4(meshFile);
	testScribeIO(soutputFile, meshFile);
}

void test_prism_CENTER()
{
	int counter = 0;  // Initialize counter to 0.
	int numTIMEs = 0; // Variable for user to enter the amount of TIMEs.

	std::string file1 = "1_elev.61";
	std::string file2 = "1_salt.70";

	SCHISMFile10 *f2 = new SCHISMFile10(file2);
	SCHISMFile10 *f2_1 = new SCHISMFile10(file2);

	SCHISMVar10 *zVarPtr = f2->get_var("Mesh2_surface");
	SCHISMVar10 *zVarPtr1 = f2_1->get_var("Mesh2_surface");

	zVarPtr->set_cur(0);
	zVarPtr1->set_cur(1);

	float *zb = new float[2];

	zVarPtr->get(zb);
	cout << "salt from first SCHISMFILE at first element:\n";
	for (int i = 0; i < 2; i++)
		cout << zb[i] << " ";

	zVarPtr1->get(zb);
	cout << "salt from second SCHISMFILE at second element:\n";
	for (int i = 0; i < 2; i++)
		cout << zb[i] << " ";

	cout << "\n";

	SCHISMVar10 *sVarPtr = f2->get_var("salt");
	float *sb = new float[10];
	sVarPtr->set_cur(0);
	sVarPtr->get(sb);
	for (int i = 0; i < 10; i++)
		cout << sb[i] << " ";

	cout << "\n";

	zVarPtr->set_cur(1);
	zVarPtr->get(zb);
	for (int i = 0; i < 2; i++)
		cout << zb[i] << " ";

	cout << "\n";

	sVarPtr->set_cur(1);
	sVarPtr->get(sb);
	for (int i = 0; i < 10; i++)
		cout << sb[i] << " ";

	cout << "\n";

	delete zb;
	delete sb;

	delete f2;

	delete f2_1;
	SCHISMFile10 *f3 = new SCHISMFile10(file2);

	delete f3;
}

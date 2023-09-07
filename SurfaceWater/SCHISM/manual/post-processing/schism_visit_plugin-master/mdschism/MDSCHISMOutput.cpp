
#include "MDSCHISMOutput.h"
#include "ncFloat.h"
#include "ncDouble.h"
#include "ncDouble.h"
#include "ncInt.h"
#include "ncShort.h"
#include "ncChar.h"

#include <sstream>
#include <vector>
#include <algorithm> 
#include <cmath>



MDSchismOutput::MDSchismOutput(const std::string a_outputFile,const std::string a_local_mesh_file):SCHISMFile10(a_outputFile),
	                                                                                                 m_local_mesh_file(a_local_mesh_file),
 																									 m_face_bottom(NULL),
                                                                                                     m_node_bottom(NULL),
                                                                                                     m_edge_bottom(NULL),
	                                                                                                 m_prism_bottom(NULL),
																									 m_face_nodes(NULL),
																									 m_side_nodes(NULL),
                                                                                                     m_face_bottom_time_id(-1),
                                                                                                     m_node_bottom_time_id(-1),
                                                                                                     m_edge_bottom_time_id(-1)
{
	//debug1 << "in nc file to create nc file "<<a_outputFile<<"\n";
	m_outputNcFilePtr=new NcFile(a_outputFile.c_str(),NcFile::ReadOnly);
	//debug1 << "in nc file done create nc file\n";
	if (m_outputNcFilePtr->is_valid()==0)
	{
		m_is_valid=false;
		throw SCHISMFileException10( a_outputFile+" is not a valid NC file\n");
	}
	else
	{
		m_is_valid=true;
	}

	//source is not available in uncombined files
	//std::string source_id="source";
	//bool find_source_att=false;
	//for(int i=0;i<m_outputNcFilePtr->num_atts();i++)
	//{
    //   NcAtt * temp_ptr=m_outputNcFilePtr->get_att(i);
	//   std::string att_id = temp_ptr->name();
	//   if (att_id==source_id)
	//   {
	//	  find_source_att=true;
	//	  std::string source = temp_ptr->as_string(0);
	//     if ((source.find("SCHISM model output")) == std::string::npos)
	//      {
	//	      throw SCHISMFileException10( a_outputFile+"is not a valid SCHSIM NC output file\n");
	//      }
	//  }
	//}


	//if (!(find_source_att))
	//{
	//	throw SCHISMFileException10( a_outputFile+"is not a valid SCHSIM NC output file\n");
	//}
	
	
	 
	//debug1 << "in nc file to begin load dim var\n";
	this->load_dim_var();
	//debug1 << "in nc file to begin fill bottom var\n";
	this->fill_bottom();
	
}

MDSchismOutput::~MDSchismOutput()
{
 
   if(m_total_num_vars)
	{
		for(int ivar=0;ivar<m_total_num_vars;ivar++)
		{
			delete m_variables[ivar];
		}
		delete [] m_variables;
	}
	if(m_total_num_dims)
	{
		for(int idim=0;idim<m_total_num_dims;idim++)
		{
			delete m_dimensions[idim];
		}
		delete [] m_dimensions;
	}

	if (m_face_bottom)
	{
		delete m_face_bottom;
	 }
    if (m_node_bottom)
	{
		delete m_node_bottom;
	}
  
	if(m_edge_bottom)
	{
		delete m_edge_bottom;
	}
 
	if (m_prism_bottom)
	{
		delete m_edge_bottom;
	}
    if(m_side_nodes)
	{
		delete m_side_nodes;
	}
	 
    if(m_face_nodes)
	{
		delete m_face_nodes;
	}
	close();
	
}
void  MDSchismOutput::close()
{
  
 if((m_outputNcFilePtr)&&(m_outputNcFilePtr->is_valid()))
 {
   m_outputNcFilePtr->close();
 }
}


int  MDSchismOutput::global_att_as_int(const std::string& a_att_name) const
{
	int num_att = m_outputNcFilePtr->num_atts();
	for(int i=0;i<num_att;i++)
	{
		NcAtt* a_att = m_outputNcFilePtr->get_att(i);
		std::string att_name =a_att->name();
		if (att_name==a_att_name)
		{
			return a_att->as_int(0);
		}
	}
	
	return 0;
}

std::string MDSchismOutput::global_att_as_string(const std::string& a_att_name) const
{
	int num_att = m_outputNcFilePtr->num_atts();
	for(int i=0;i<num_att;i++)
	{
		NcAtt* a_att = m_outputNcFilePtr->get_att(i);
		std::string att_name =a_att->name();
		if (att_name==a_att_name)
		{
			return a_att->as_string(0);
		}
	}
	
	return "";
}



void   MDSchismOutput::fill_bottom()
{

   ifstream*    localFileStream = new ifstream(m_local_mesh_file.c_str()); 
   //debug1 << "in nc file done create local mesh stream\n";
   if (!localFileStream->good())
   {
       throw SCHISMFileException10("not a valid file "+ m_local_mesh_file);
   }
    std::string  lineTemp;
  
    std::getline(*localFileStream,lineTemp);
	std::getline(*localFileStream,lineTemp);
	//std::getline(*localFileStream,lineTemp);
	//std::stringstream neaStream(lineTemp);
	long number_element,number_node,number_side;
	(*localFileStream)>> number_element;
	

	for(long iEle=0;iEle<number_element;iEle++)
		{
		 long t1,t2;
		(*localFileStream)>>t1>>t2;
		 
		}
	
	//std::getline(*localFileStream,lineTemp);
	//std::stringstream npaStream(lineTemp);
	(*localFileStream)>> number_node;
    

	for(long iNode=0;iNode<number_node;iNode++)
		{
		 
		 long t1,t2;
		(*localFileStream)>>t1>>t2;
		 
		}
		
    //std::getline(*localFileStream,lineTemp);
	//std::stringstream nsaStream(lineTemp);
	(*localFileStream)>> number_side;
   

	for(long iSide=0;iSide<number_side;iSide++)
		{
		 
		 long t1,t2;
		(*localFileStream)>>t1>>t2;
		 
		}
   
    std::string temp;
	//std::getline(*localFileStream,lineTemp);
	(*localFileStream)>>temp;
    //std::getline(*localFileStream,lineTemp);
	int year,month,day;
    double time,tzone;
	(*localFileStream) >> year >> month >> day >> time>> tzone;
    //std::getline(*localFileStream,lineTemp);
    //std::stringstream vStream(lineTemp);
    int number_layer;
	int nrec, nspool,kz,ics;
	double v2,h0,hs,hc,thetab,thetaf;
	(*localFileStream) >> nrec >> v2 >> nspool >> number_layer >> kz >> h0>>hs>>hc>>thetab>>thetaf>>ics;
	
    std::getline(*localFileStream,lineTemp); //sigma stream
	double zt,sigmat;
	for(int i=1;i<kz;i++)
		(*localFileStream)>>zt;
    for(int i=1;i<number_layer-kz+2;i++)
		(*localFileStream)>>sigmat;

	long t1,t2;
    (*localFileStream)>>t1>>t2;
	//read in x,y,dp, kpb00
	
	
	if(!(m_node_bottom))
    {
	  m_node_bottom=new int[number_node];
    }
	if(!(m_face_bottom))
    {
	  m_face_bottom=new int[number_element];
    }
	if(!(m_edge_bottom))
    {
	  m_edge_bottom=new int[number_side];
    }
	
	int stepsize =  MeshConstants10::MAX_NUM_NODE_PER_CELL+1;
	
	if(!m_face_nodes)
	{
	   m_face_nodes = new long   [stepsize*number_element];
	}
	
    if(!m_side_nodes)
	{
	   m_side_nodes = new long   [2*number_side];
	}
	
	for(long iNode=0;iNode<number_node;iNode++)
	{	
		//std::getline(*localFileStream,lineTemp);
		//std::stringstream coordStream(lineTemp);
		double x,y,dp;
		(*localFileStream)>>x>>y>>dp>>m_node_bottom[iNode];
	}
	// read in face nodes
	for(long iEle=0;iEle<number_element;iEle++)
	{
		long numNode;
		(*localFileStream)>>numNode;
		m_face_nodes[0+iEle*stepsize]=numNode;
		int min_bottom=1; //schism bottom level is 1
		for(long iNode=0;iNode<numNode;iNode++)
		{
			int node_id;
			(*localFileStream)>>node_id;
		    m_face_nodes[iNode+1+iEle*stepsize]=node_id;
			if(iNode==0)
			{
				min_bottom=m_node_bottom[node_id];
			}
			else
			{
				if(min_bottom>m_node_bottom[node_id])
				{
					min_bottom=m_node_bottom[node_id];
				}
			}
		}
		m_face_bottom[iEle]=min_bottom;
		
	}
	
	//read in side nodes
	for(long iSide=0;iSide<number_side;iSide++)
	{
		long sideid,node1,node2;
		(*localFileStream)>>sideid>>node1>>node2;
		m_side_nodes[0+iSide*2]=node1-1;
		m_side_nodes[1+iSide*2]=node2-1;
		int min_bottom=m_node_bottom[node1-1]; //schism bottom level is 1
		if (min_bottom>m_node_bottom[node2-1])
		{
			min_bottom=m_node_bottom[node2-1];
		}
		m_edge_bottom[iSide]=min_bottom;
	}

	localFileStream->close();
	delete localFileStream;
}

void   MDSchismOutput::get_node_bottom(int* a_node_bottom,const int& a_time)
{
   NcDim * dimNodePtr      = m_outputNcFilePtr->get_dim(MeshConstants10::DIM_MESH_NODES.c_str());
   long numMeshNodes=0;
   numMeshNodes = dimNodePtr->size(); 

   for(long i=0;i<numMeshNodes;i++)
   {
	   a_node_bottom[i]=m_node_bottom[i];
   }

}
void   MDSchismOutput::get_face_bottom(int* a_face_bottom,const int& a_time)
{
   NcDim * dimFacePtr      = m_outputNcFilePtr->get_dim(MeshConstants10::DIM_MESH_FACES.c_str());
   long numMeshFaces=0;
   numMeshFaces = dimFacePtr->size(); 

   for(long i=0;i<numMeshFaces;i++)
   {
	   a_face_bottom[i]=m_face_bottom[i];
   }
}

void   MDSchismOutput::get_prism_bottom(int* a_prism_bottom, const int& a_time)
{
	NcDim * dimFacePtr = m_outputNcFilePtr->get_dim(MeshConstants10::DIM_MESH_FACES.c_str());
	long numMeshFaces = 0;
	numMeshFaces = dimFacePtr->size();

	for (long i = 0; i < numMeshFaces; i++)
	{
		a_prism_bottom[i] = m_prism_bottom[i];
	}
}

void  MDSchismOutput::get_edge_bottom(int* a_edge_bottom,const int& a_time)
{
   NcDim * dimEdgePtr      = m_outputNcFilePtr->get_dim(MeshConstants10::DIM_MESH_EDGES.c_str());
   long numMeshEdges=0;
   numMeshEdges = dimEdgePtr->size(); 

   for(long i=0;i<numMeshEdges;i++)
   {
	   a_edge_bottom[i]=m_edge_bottom[i];
   }
}

 bool    MDSchismOutput::update_bottom_index(const int& a_time)
 {
	 bool bottom_changed =false;
	 bottom_changed=update_node_bottom(a_time);
	 if (bottom_changed)
	 {
	    update_ele_bottom(a_time,m_node_bottom);
	    update_edge_bottom(a_time,m_node_bottom);
	 }
	 return bottom_changed;
 }
bool   MDSchismOutput::update_node_bottom(const int& a_time)
{
  
  NcDim * dimNodePtr      = m_outputNcFilePtr->get_dim(MeshConstants10::DIM_MESH_NODES.c_str());
  long numMeshNodes=0;
  numMeshNodes = dimNodePtr->size(); 
  NcDim * dimLayerPtr      = m_outputNcFilePtr->get_dim(MeshConstants10::DIM_LAYERS.c_str());
  long nLayers=0;
  nLayers = dimLayerPtr->size(); 

  NcVar * ncvar = m_outputNcFilePtr->get_var(MeshConstants10::ZCOORD.c_str());

  float missing_val = (ncvar->get_att("missing_value"))->as_float(0);

  float * zcor=new float[numMeshNodes*nLayers];

  long current[3];
  current[0]=a_time;
  current[1]=0;
  current[2]=0;
  long count[3];
  count[0]=1;
  count[1]=numMeshNodes;
  count[2]=nLayers;

  ncvar->set_cur(current);
  ncvar->get(zcor,count[0],count[1],count[2]);
  m_node_bottom_time_id=a_time;
  
  int  bottom_id=nLayers+1;
  bool bottom_changed=false;
  
 
  for(long i=0;i<numMeshNodes;i++)
  {
	  bottom_id=nLayers+1;
	  for(int ilevel=0;ilevel<nLayers;ilevel++)
	  {
		  if(zcor[i*nLayers+ilevel]!=missing_val)
		  {
			  bottom_id=ilevel+1;
			  break;
		  }
	  }

    if(bottom_id!=m_node_bottom[i])
	{
		bottom_changed=true;
		m_node_bottom[i]=bottom_id;
	}
     
  }
  delete zcor;
  return bottom_changed;
}
void   MDSchismOutput::update_edge_bottom(const int& a_time, int* a_node_bottom)
{
  NcDim * dimFacePtr      = m_outputNcFilePtr->get_dim(MeshConstants10::DIM_MESH_FACES.c_str());
  long numMeshFaces=0;
  numMeshFaces = dimFacePtr->size(); 
  NcDim * dimNodePtr      = m_outputNcFilePtr->get_dim(MeshConstants10::DIM_MESH_NODES.c_str());
  long numMeshNodes=0;
  numMeshNodes = dimNodePtr->size(); 

  NcDim * dimEdgePtr      = m_outputNcFilePtr->get_dim(MeshConstants10::DIM_MESH_EDGES.c_str());
  long numMeshEdges=0;
  numMeshEdges = dimEdgePtr->size(); 




  if(!(m_edge_bottom))
  {
	  m_edge_bottom=new int[numMeshEdges];
  }

  for(long iedge=0;iedge<numMeshEdges;iedge++)
  {
	  long node1 = m_side_nodes[iedge*2];
	  long node2 = m_side_nodes[iedge*2+1]; //node id is 1 based
	  m_edge_bottom[iedge]=a_node_bottom[node1];
	  if (a_node_bottom[node2]<a_node_bottom[node1])
	  {
		   m_edge_bottom[iedge]=a_node_bottom[node2];
	  }

  }

  m_edge_bottom_time_id=a_time;
 
}
void   MDSchismOutput::update_ele_bottom(const int& a_time,int* a_node_bottom)
{
   
  NcDim * dimFacePtr      = m_outputNcFilePtr->get_dim(MeshConstants10::DIM_MESH_FACES.c_str());
  long numMeshFaces=0;
  numMeshFaces = dimFacePtr->size(); 
  NcDim * dimNodePtr      = m_outputNcFilePtr->get_dim(MeshConstants10::DIM_MESH_NODES.c_str());
  long numMeshNodes=0;
  numMeshNodes = dimNodePtr->size(); 
 

  if(!(m_face_bottom))
  {
	  m_face_bottom=new int[numMeshFaces];
  }

  for(long iface=0;iface<numMeshFaces;iface++)
  {
	  int num_point_in_face = m_face_nodes[iface*(MeshConstants10::MAX_NUM_NODE_PER_CELL+1)];
	  int min_node_bottom= MeshConstants10::BIG_LEVEL;
	  for(int inode=0;inode<num_point_in_face;inode++)
	  {
		  long node_id = m_face_nodes[iface*(MeshConstants10::MAX_NUM_NODE_PER_CELL+1)+inode+1]; //node id is 1 based
		  int node_bottom=a_node_bottom[node_id];
		  if (node_bottom<min_node_bottom)
		  {
			  min_node_bottom=node_bottom;
		  }
	  }
	  m_face_bottom[iface]=min_node_bottom;

  }
  m_face_bottom_time_id=a_time;
 
}

void   MDSchismOutput::set_prism_bottom(const int& a_time, int* a_prism_bottom)
{
	update_prism_bottom(a_time, a_prism_bottom);
}

void   MDSchismOutput::update_prism_bottom(const int& a_time, int* a_prism_bottom)
{

	NcDim * dimFacePtr = m_outputNcFilePtr->get_dim(MeshConstants10::DIM_MESH_FACES.c_str());
	long numMeshFaces = 0;
	numMeshFaces = dimFacePtr->size();
	NcDim * dimNodePtr = m_outputNcFilePtr->get_dim(MeshConstants10::DIM_MESH_NODES.c_str());
	long numMeshNodes = 0;
	numMeshNodes = dimNodePtr->size();


	if (!(m_prism_bottom))
	{
		m_prism_bottom = new int[numMeshFaces];
	}

	for (long iface = 0; iface < numMeshFaces; iface++)
	{
		
		m_prism_bottom[iface] = a_prism_bottom[iface];

	}
	m_prism_bottom_time_id = a_time;

}



bool  MDSchismOutput::load_dim_var()
{
	int kz = 1;
	
   
	int num_dim = m_outputNcFilePtr->num_dims();
	m_dimensions = new SCHISMDim10 * [num_dim];
	m_total_num_dims = num_dim;
	for(int idim=0;idim<num_dim;idim++)
	{
		NcDim * a_dim = m_outputNcFilePtr->get_dim(idim);
		
		SCHISMDim10 * wrapped_dim;

		if (a_dim->name() == MeshConstants10::DIM_MESH_FACE_NODES)
		{
		  wrapped_dim = newSCHISMDim(a_dim->name(),
			                                   idim,
											   a_dim->size()+1);
		}
		else
		{
		  wrapped_dim = newSCHISMDim(a_dim->name(),
			                                   idim,
											   a_dim->size());
		}
		m_dimensions[idim]= wrapped_dim;
	}
	
	
	std::map<std::string, int>  var_name_added;

	int num_var = m_outputNcFilePtr->num_vars();
	 m_variables  = new SCHISMVar10 * [num_var];
	for(int ivar=0;ivar<num_var;ivar++)
	{
		NcVar * a_var =m_outputNcFilePtr->get_var(ivar);
		std::string var_name = a_var->name();
		MDSchismOutputVar * schism_var        = new MDSchismOutputVar(var_name);
		schism_var->fill_ncVar(a_var);
		schism_var->m_schismfilePtr = this;
		int num_dim_var = a_var->num_dims();
		
		
		// filled global dim id in ncfile
		for(int i_dim_var=0;i_dim_var<num_dim_var;i_dim_var++)
		{
			NcDim * dim_var = a_var->get_dim(i_dim_var);
			std::string dim_name = dim_var->name();
			for(int i_dim_file=0;i_dim_file<num_dim;i_dim_file++)
			{
				if(m_dimensions[i_dim_file]->name() == dim_name)
				{
					schism_var->add_dimension(i_dim_file);
					break;
				}
			}
		}
		
		// fill att
		int num_att_var = a_var->num_atts();
		
		for(int i_att_var=0;i_att_var<num_att_var;i_att_var++)
		{
			NcAtt * att_var = a_var->get_att(i_att_var);
		    SCHISMAtt10 * schism_att = new SCHISMAtt10(att_var->name());
			NcType type = att_var->type();
			if (type == ncChar) 
			{
				schism_att->add_string_value(att_var->as_string(0));

			}
			else if ((type == ncShort) || (type ==ncInt))
			{
				schism_att->add_int_value(att_var->as_int(0));
			}
			else if (type==ncFloat)
			{
				schism_att->add_float_value(att_var->as_float(0));
			}
			else if (type==ncDouble)
			{
				schism_att->add_double_value(att_var->as_double(0));
			}
			schism_var->add_att(schism_att);

			std::string nc_att_name(att_var->name());

			//if ((!(nc_att_name.compare(MeshConstants10::CENTER)))||(!(nc_att_name.compare(MeshConstants10::LOCATION))))
			//{
			//	schism_var->set_horizontal_center((att_var->as_string(0)));
			//}
			//else if (!(nc_att_name.compare(MeshConstants10::LAYER_CENTER)))
			//{
			//	schism_var->set_vertical_center(att_var->as_string(0));
			//}

			if (!(nc_att_name.compare(MeshConstants10::I23D)))
			{
				int i23d = att_var->as_int(0);
				if (i23d <= 3)
				{
					schism_var->set_horizontal_center(MeshConstants10::NODE);
				}
				else if (i23d <= 6)
				{
					schism_var->set_horizontal_center(MeshConstants10::ELEM);
				}
				else if (i23d <= 9)
				{
					schism_var->set_horizontal_center(MeshConstants10::EDGE);
				}
				else
				{
					throw SCHISMVarException10("i23d is not a valid\n");
				}

				if (!(i23d % 3))
				{
					schism_var->set_vertical_center(MeshConstants10::HALF_LAYER);
				}
				else
				{
					schism_var->set_vertical_center(MeshConstants10::FULL_LAYER);
				}
			}
		
		}

		
		
		int var_id            = m_total_num_vars;
        m_variables[var_id]   = schism_var;
		var_name_added[var_name]=var_id;
        m_total_num_vars++; 
	}
	return true;
}


 

MDSchismOutputVar::~MDSchismOutputVar()
{

}

bool  MDSchismOutputVar::put_a_float(const float&    a_value,
                                int  *    a_dim1Loc)
{
	return false;
}

void  MDSchismOutputVar::fill_ncVar(NcVar * a_nc_var)
{
	m_ncVar[0] = a_nc_var;
	m_num_component=1;
	
	for(int iatt=0;iatt<a_nc_var->num_atts();iatt++)
	{
		NcAtt* att_ptr=0;

		att_ptr =a_nc_var->get_att(iatt);

		string att_name = att_ptr->name(); 
		
		if (!att_name.compare("ivs")) //this is the label for number of vector components
		{
			m_num_component=att_ptr->as_int(0);
			break;
		}

		delete att_ptr;
	}


}

 void      MDSchismOutputVar::fill_current_bottom(int * a_kbp00) 
 {
   if(m_horizontal_center==MeshConstants10::NODE)
   {
	    m_schismfilePtr->get_node_bottom(a_kbp00,m_current_record);
   }
   else if(m_horizontal_center==MeshConstants10::EDGE)
   {
	    m_schismfilePtr->get_edge_bottom(a_kbp00,m_current_record);
   }
   else 
   {
	   if (m_vertical_center == MeshConstants10::HALF_LAYER)
	   {
		   m_schismfilePtr->get_prism_bottom(a_kbp00, m_current_record);
	   }
	   else
	   {
		   m_schismfilePtr->get_face_bottom(a_kbp00, m_current_record);
	   }
   }
 }

bool  MDSchismOutputVar::get(float *     a_buffer) 
{

 int dataSize = computeDataNumPerTIMEStep();

 if(m_data_cached)
 {
   //for(int idata=0;idata<dataSize;idata++)
   //{
    //  a_buffer[idata] = m_float_cache[idata];
   //}
   //return true;
   return get_float_cache(a_buffer);
 }
  return load_from_file<float>(a_buffer);
}

bool  MDSchismOutputVar::get(double *     a_buffer) 
{

 int dataSize = computeDataNumPerTIMEStep();
  return load_from_file<double>(a_buffer);
}


bool  MDSchismOutputVar::get(int *     a_buffer)
{

 int dataSize = computeDataNumPerTIMEStep();
 //debug1 << "data size is " << dataSize<<"\n";
 if(m_data_cached)
 {
   //for(int idata=0;idata<dataSize;idata++)
   //{
   //   a_buffer[idata] = m_int_cache[idata];
   //}
   //return true;
   return get_int_cache(a_buffer);
 }
  
 return load_from_file<int>(a_buffer);
}


  //  get data from cache
bool   MDSchismOutputVar::get_int_cache(int *       a_buffer)  
{
   long dataSize = computeDataNumPerTIMEStep();
   for(long idata=0;idata<dataSize;idata++)
   {
      a_buffer[idata] = m_int_cache[idata];
   }
   return true;
}

bool   MDSchismOutputVar::get_long_cache(long *       a_buffer) 
{
   long dataSize = computeDataNumPerTIMEStep();
   for(long idata=0;idata<dataSize;idata++)
   {
      a_buffer[idata] = m_long_cache[idata];
   }
   return true;
}

bool   MDSchismOutputVar::get_float_cache(float *     a_buffer) 
{
   long dataSize = computeDataNumPerTIMEStep();
   for(long idata=0;idata<dataSize;idata++)
   {
      a_buffer[idata] = m_float_cache[idata];
   }
   return true;
}



bool  MDSchismOutputVar::get(long *     a_buffer) 
{

 long dataSize = computeDataNumPerTIMEStep();

 if(m_data_cached)
 {
  // for(long idata=0;idata<dataSize;idata++)
  // {
  //    a_buffer[idata] = m_long_cache[idata];
  // }
  // return true;
	return get_long_cache(a_buffer);
 }
  
  return load_from_file<long>(a_buffer);
}

void  MDSchismOutputVar::set_cur(const int& a_time_record)
{
	m_current_record = a_time_record;
}




MDSchismOutputVar::MDSchismOutputVar(const std::string& a_varName):SCHISMVar10(a_varName)
{
	  m_num_component = 0;
}
MDSchismOutputVar::MDSchismOutputVar():SCHISMVar10()
{
	  m_num_component=0;
}

int  MDSchismOutputVar::computeDataNumPerTIMEStep() const
{
	int dataSize =1;

  for(int iDim=0;iDim<m_num_dim;iDim++)
  { 
     SCHISMDim10 * aDim = m_schismfilePtr->get_dim(m_dimensions[iDim]);
  
     if(!(aDim->name()=="TIME"))
      {
        dataSize *= aDim->size();
      }
      else if (m_name.compare("TIME")==0)
     {
        dataSize *= aDim->size();
     }
    else
     {
     }
  }
  return dataSize;
}





 
   



  

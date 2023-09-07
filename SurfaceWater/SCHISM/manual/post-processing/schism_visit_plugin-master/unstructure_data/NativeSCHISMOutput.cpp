#include "NativeSCHISMOutput.h"
#include "MeshConstants.h"
#include "SCHISMFileUtil.h"
#include <sstream>
#include <stdlib.h>
#include <algorithm>


const int DATA_NUM_BYTE = 4;
const int MAX_CELL_NODE = MeshConstants::MAX_NUM_NODE_PER_CELL;
const int         DEFAULT_INT_VALUE         = -9999;
const float       DEFAULT_FLOAT_VALUE       = -9999.0;
const std::string DEFAULT_STR_VALUE         = "error";

NativeSchismOutputVar::NativeSchismOutputVar(const std::string& a_varName):SCHISMVar(a_varName)
{  
}


NativeSchismOutputVar::NativeSchismOutputVar():SCHISMVar()
{ 
}

void NativeSchismOutputVar::set_cur(const int& a_record)
{
  
  //TIME dimension must be first dimension
  SCHISMDim * TIMEDim = m_schismfilePtr->get_dim(m_dimensions[0]);
  
  //move filestream to specified TIME block
  if(TIMEDim->name()==MeshConstants::DIM_TIME)
  {
    m_schismfilePtr->set_cur(a_record,m_offsetInDataBlock);
  }
 
}

bool NativeSchismOutputVar:: put_a_float(const float&    a_value,
                                         int  *    a_dimLoc)
{
  int dataSize = computeDataNumPerTIMEStep();

  if(!m_float_cache)
  {
    m_float_cache = new float [dataSize];
  }

  int dimSize     = num_dims();
  int bufferIndex = 0;
  for(int iDim=0;iDim<dimSize;iDim++)
   {
     int temp            = bufferIndex;
     int nextDimSize = 1;
     if(iDim < (dimSize-1))
     {
       SCHISMDim * nextDim = m_schismfilePtr->get_dim(m_dimensions[iDim+1]);
       nextDimSize        = nextDim->size();
     }
     bufferIndex   = (temp+a_dimLoc[iDim])*nextDimSize;

   }  
  m_float_cache[bufferIndex] = a_value;
  return true;
}


bool NativeSchismOutputVar::get(int * a_buffer) const
{
  int dataSize = computeDataNumPerTIMEStep();

  if(m_data_cached)
   {
     for(int idata=0;idata<dataSize;idata++)
      {
        a_buffer[idata] = m_int_cache[idata];
      }
   }
  else
  {
     union
     {
         int* intArray;
         char * cBuffer;
     } intArrayReader;
    
    intArrayReader.cBuffer = new char [dataSize*DATA_NUM_BYTE]; 
    m_schismfilePtr->read(intArrayReader.cBuffer,dataSize*DATA_NUM_BYTE);

    for(int idata=0;idata<dataSize;idata++)
      {
        a_buffer[idata] = intArrayReader.intArray[idata];
      }
    
    delete  intArrayReader.cBuffer;
  }
  return true;
}





bool NativeSchismOutputVar::get(float * a_buffer) const
{
  int dataSize = computeDataNumPerTIMEStep();

  if(m_data_cached)
   {
     for(int idata=0;idata<dataSize;idata++)
      {
        a_buffer[idata] = m_float_cache[idata];
      }
   }
  else
  {
     union
     {
         float*   floatArray;
         char * cBuffer;
     } floatArrayBuffer;
     
    bool hasLayer = hasVerticalLayerDimension();

    if(!hasLayer)
    {   
 
      floatArrayBuffer.cBuffer = new char [dataSize*DATA_NUM_BYTE]; 
      m_schismfilePtr->read(floatArrayBuffer.cBuffer,dataSize*DATA_NUM_BYTE);

      for(int idata=0;idata<dataSize;idata++)
      {
        a_buffer[idata] = floatArrayBuffer.floatArray[idata];
      }
    
      delete  floatArrayBuffer.cBuffer;
    }
    else
    {
      int staggeredDataSize = computeDataNumPerTIMEStepWitHStaggeredLayers();
      floatArrayBuffer.cBuffer = new char [staggeredDataSize*DATA_NUM_BYTE]; 
      m_schismfilePtr->read(floatArrayBuffer.cBuffer,staggeredDataSize*DATA_NUM_BYTE);
      // nodeDim is dim 1
      SCHISMDim * nodeDim = m_schismfilePtr->get_dim(m_dimensions[1]);
      int totalNodeNum = nodeDim->size();
      
      int comSize =1;
      if (m_num_dim>3)
      {
        SCHISMDim * comDim = m_schismfilePtr->get_dim(m_dimensions[3]);
        comSize = comDim->size();
      }
      SCHISMDim * layerDim = m_schismfilePtr->get_dim(m_dimensions[2]);
      int  numLayer = layerDim->size();
	  std::string level_center = (this->m_schismfilePtr)->level_center();

      int read_buffer_index = 0;
	  int out_buffer_index =0;
      int * kbp00 = new int [totalNodeNum];
	  fill_bottom(kbp00);
      for(int iNode=0;iNode<totalNodeNum;iNode++)
      {
		  int valid_bottom_layer = std::max(1,kbp00[iNode]);
          for(int iLayer = valid_bottom_layer; iLayer<numLayer+1;iLayer++)
          {
             for(int iComponent = 0 ;iComponent<comSize; iComponent++)
                {
				   //int numDataLayer=numLayer-max(1,m_kbp00[iNode])+1;
                   //int outDataIndex = iNode*numDataLayer*comSize + (iLayer -1) * comSize + iComponent;
					// remove extra bottom data for half level dataset
				   if ((level_center != MeshConstants::HALF_LAYER) || ((level_center == MeshConstants::HALF_LAYER)&&(iLayer>valid_bottom_layer)))
				   {
                      a_buffer[out_buffer_index] = floatArrayBuffer.floatArray[read_buffer_index];
					  out_buffer_index++;
				   }
                   read_buffer_index++;
                 }
          }  
       } 
      delete kbp00;
      delete  floatArrayBuffer.cBuffer;
   }
 }
  return true;
}




void NativeSchismOutputVar::setOffsetInDataBlock(const int& a_offset)
{
   m_offsetInDataBlock = a_offset;
}

NativeSchismOutputVar::~NativeSchismOutputVar() 
{
  
}


NativeSchismOutput::NativeSchismOutput(const std::string a_SCHISMOutputFile):SCHISMFile(a_SCHISMOutputFile),
                                                          m_numByte(4),
                                                          m_dataBlockBeginOffset(0),
                                                          m_dataBlockLength(0)
{
   m_schismOutputFileStream = new ifstream(a_SCHISMOutputFile.c_str(),ios::binary); 
   
   if(m_schismOutputFileStream->is_open())
   {
     
   }
   else
   {
      m_is_valid = false;
	  return;
   }


   // get length of file:
   m_schismOutputFileStream->seekg (0, ios::end);
   m_fileLength =  m_schismOutputFileStream->tellg();
   
   m_schismOutputFileStream->seekg (0, ios::beg);
   
   m_varLongNameMap["hvel"] = "hvel";
   m_varLongNameMap["elev"] = "surface_elevation";
   m_varLongNameMap["salt"] = "salt";
   m_varLongNameMap["temp"] = "temp";
   m_is_valid                = load_dims_Vars();
  

}

NativeSchismOutput::~NativeSchismOutput()
{
  close();
  if(m_total_num_vars)
  {
	 for(int ivar=0;ivar<m_total_num_vars;ivar++)
	 {
	 	delete m_variables[ivar];
	 }
	 delete m_variables;
  }
  if(m_total_num_dims)
  {
	 for(int idim=0;idim<m_total_num_dims;idim++)
	 {
	 	delete m_dimensions[idim];
	 }
	 delete m_dimensions;
  }
}



void   NativeSchismOutput::set_cur(const int& a_step,
                          const int& a_extraOffset)
{

  long long  pos = (long long)(m_dataBlockBeginOffset) +
             (long long)(a_step)* (long long)(m_dataBlockLength) +
             (long long) (a_extraOffset);

  m_schismOutputFileStream->seekg( pos,ios::beg);

  if ( ( m_schismOutputFileStream->rdstate() & std::ifstream::failbit ) != 0 )
  {
	std::stringstream ss;
	ss<<"fail to move filestream loc when setting current TIME step "
	<<a_step<<" \n";
    throw SCHISMFileException(ss.str());
  }
}

void NativeSchismOutput::close()
{

 if(m_schismOutputFileStream)
 {
   if(m_schismOutputFileStream->is_open())
     {
        m_schismOutputFileStream->close();
     }
   delete  m_schismOutputFileStream;
 }
 
}

bool    NativeSchismOutput::read(char * a_buffer, const int& a_numByte)
{
  
   m_schismOutputFileStream->read(a_buffer,a_numByte);
   return true;
}

bool NativeSchismOutput::load_dims_Vars()
{
 m_schismOutputFileStream->read(m_data_format,48);
 m_schismOutputFileStream->read(m_data_description,48);  
 m_schismOutputFileStream->read(m_start_time,48);
 m_schismOutputFileStream->read(m_var_nm,48);
 m_schismOutputFileStream->read(m_var_dim,48);

 struct Header
          {
             int        numberRecord ;  
             float      dtout        ;
             int        nspool       ;
             int        ivs          ;
             int        i23d         ;
             int        nvrt         ;
             int        kz           ;
             float      h0           ;
             float      h_s          ;
             float      h_c          ;
             float      theta_b      ;
             float      theta_f      ;
           };
    
  union  {
           char*    cBuffer;
           Header*  header;
         } readHeaderBuffer;

  
  readHeaderBuffer.cBuffer = new char [12*DATA_NUM_BYTE];
  m_schismOutputFileStream->read(readHeaderBuffer.cBuffer,12*DATA_NUM_BYTE);

  int nvrt   =  readHeaderBuffer.header->nvrt;
  int kz     =  readHeaderBuffer.header->kz;
  int i23d   =  readHeaderBuffer.header->i23d;
  int ivs    =  readHeaderBuffer.header->ivs;
  int nrec   =  readHeaderBuffer.header->numberRecord;

   float HS       =  readHeaderBuffer.header->h_s;
   float Hc       =  readHeaderBuffer.header->h_c;
   float THETAB   =  readHeaderBuffer.header->theta_b;
   float THETAF   =  readHeaderBuffer.header->theta_f;

  delete readHeaderBuffer.cBuffer;


  
  union {
           float *       values      ;
           char  *       cBuffer     ;
        } floatArrayBuffer;

   floatArrayBuffer.cBuffer    = new char [nvrt*DATA_NUM_BYTE] ;
   m_schismOutputFileStream->read(floatArrayBuffer.cBuffer,(kz-1)*DATA_NUM_BYTE) ;
   m_schismOutputFileStream->read(floatArrayBuffer.cBuffer,(nvrt-kz+1)*DATA_NUM_BYTE);

   int nSigmaDim            = m_total_num_dims;
   SCHISMDim * sigmaDim      = newSCHISMDim(MeshConstants::DIM_SIGMA_LAYERS,
                                            nSigmaDim,
                                            nvrt-kz+1);
   m_dimensions[nSigmaDim]  = sigmaDim;
   m_total_num_dims++;

   int nvDim                = m_total_num_dims;
   SCHISMDim * vDim          = newSCHISMDim(MeshConstants::DIM_LAYERS ,
                                          nvDim,
                                          nvrt);
   m_dimensions[nvDim]     = vDim;
   m_total_num_dims++;


   int nzDim                = m_total_num_dims;
   SCHISMDim * zDim          = newSCHISMDim(MeshConstants::DIM_KZ_LAYERS ,
                                            nzDim,
                                            kz-1);
   m_dimensions[nzDim]      = zDim;
   m_total_num_dims++;

   int nComponentDim        = -1; //a invalid dim id
   if(ivs>1)
   {
    nComponentDim           = m_total_num_dims;
    SCHISMDim * comDim       = newSCHISMDim(MeshConstants::DIM_VAR_COMPONENT,
                                           nComponentDim,
                                           ivs);
    m_dimensions[nComponentDim] = comDim;
    m_total_num_dims++;
   }
   
   NativeSchismOutputVar * sigmaVar        = new NativeSchismOutputVar(MeshConstants::LAYER_SCOORD);
   sigmaVar->m_schismfilePtr   = this;
   sigmaVar->add_dimension(nSigmaDim);

   SCHISMAtt * sigma_center_att = new SCHISMAtt(m_var_location_att);
   sigma_center_att->add_string_value("node");
  

   SCHISMAtt * HSAtt = new SCHISMAtt(m_hs_att);
   HSAtt->add_float_value(HS);
   SCHISMAtt * HcAtt = new SCHISMAtt(m_hc_att);
   HcAtt->add_float_value(Hc);
   SCHISMAtt * THETABAtt = new SCHISMAtt(m_thetab_att);
   THETABAtt->add_float_value(THETAB);
   SCHISMAtt * THETAFAtt = new SCHISMAtt(m_thetaf_att);
   THETAFAtt->add_float_value(THETAF);

   sigmaVar->add_att(sigma_center_att);
   sigmaVar->add_att(HSAtt);
   sigmaVar->add_att(HcAtt);
   sigmaVar->add_att(THETABAtt);
   sigmaVar->add_att(THETAFAtt);
   

   //add sigma variable 
   int sigmaVarId            = m_total_num_vars;
   m_variables[sigmaVarId]    = sigmaVar;
   m_total_num_vars++;   
 

   //fill sigma values
   sigmaVar->cache_data(floatArrayBuffer.values);
  
   union
    {
        int*   intValue;
        char*  fourChar;
     } singleIntBuffer;

    singleIntBuffer.fourChar = new char [DATA_NUM_BYTE];

    m_schismOutputFileStream->read(singleIntBuffer.fourChar,DATA_NUM_BYTE);
	//it might be number of node(61-64), ele (70) or sides (67) depends on file type
    int totalNumberOfNodes = *(singleIntBuffer.intValue);

    m_schismOutputFileStream->read(singleIntBuffer.fourChar,DATA_NUM_BYTE);

    int totalNumberOfEle   = *(singleIntBuffer.intValue);
 
    delete   floatArrayBuffer.cBuffer;
    delete   singleIntBuffer.fourChar;

    //total chars needed to be read in for x y dp 
    floatArrayBuffer.cBuffer = new char [3*DATA_NUM_BYTE];
	singleIntBuffer.fourChar = new char [DATA_NUM_BYTE];

	float * xcoord         = new float [totalNumberOfNodes];
	float * ycoord = new float [totalNumberOfNodes];
	float * depth = new float [totalNumberOfNodes];
	int * kbp00 = new int [totalNumberOfNodes];
	for(int iNode =0;iNode<totalNumberOfNodes;iNode++)
	{
		//this is x,y ,dp,they are float
		m_schismOutputFileStream->read(floatArrayBuffer.cBuffer,3*DATA_NUM_BYTE);
		//this is kbp00, has to be int
		m_schismOutputFileStream->read(singleIntBuffer.fourChar,DATA_NUM_BYTE);
		kbp00[iNode] = *singleIntBuffer.intValue; 
		xcoord[iNode]= floatArrayBuffer.values[0]; 
		ycoord[iNode]= floatArrayBuffer.values[1]; 
		depth[iNode]= floatArrayBuffer.values[2]; 
	}

  SCHISMAtt * center_att = new SCHISMAtt(m_var_location_att);

  //61-64 are node CENTERed, 65 above is face CENTER

  size_t startPos              = m_SCHISMOutputFile.find_last_of(".");
  std::string fileType = m_SCHISMOutputFile.substr(startPos+1,2);   

  int fileTypeInt      = -9999;

  try
  {
     fileTypeInt       = atoi(fileType.c_str());;
  }
  catch (...)
  {
    cout<<"file name "<<m_SCHISMOutputFile<<" don't contain SCHISM output type surfix (61-70)\n";
    return false;
  }
 
   std::string node_dim_name = MeshConstants::DIM_MESH_NODES;
   std::string node_bottom_name = MeshConstants::NODE_BOTTOM;
 
  if ((fileTypeInt==67)||(fileTypeInt==68)||(fileTypeInt==65)) //nodes of mesh defined by original mesh side center
   {
	  node_dim_name = MeshConstants::DIM_MESH_EDGES;
	  node_bottom_name = MeshConstants::EDGE_BOTTOM;
	  m_data_center = MeshConstants::EDGE;
	  if(fileTypeInt==68)
	  {
		  m_layer_center = MeshConstants::HALF_LAYER;
	  }
   }
   else if ((fileTypeInt==70))
   {
      node_dim_name = MeshConstants::DIM_MESH_FACES;
	  node_bottom_name = MeshConstants::FACE_BOTTOM;
	  m_data_center = MeshConstants::ELEM;
	  m_layer_center = MeshConstants::HALF_LAYER;
   }
  else if ((fileTypeInt>=65)&&(fileTypeInt<=69))
   {
       node_dim_name = MeshConstants::DIM_MESH_FACES;
	   node_bottom_name = MeshConstants::FACE_BOTTOM;
	   m_data_center =MeshConstants::ELEM;
   }
 
   
   // add node dim
   int nNodeDim                = m_total_num_dims;
   SCHISMDim * nodeDim          = newSCHISMDim(node_dim_name,
                                              nNodeDim,
                                              totalNumberOfNodes);
   m_dimensions[nNodeDim]      = nodeDim;
   m_total_num_dims++;


   NativeSchismOutputVar * bottom_var        = new NativeSchismOutputVar(node_bottom_name);
   bottom_var->add_dimension(nNodeDim);
   bottom_var->m_schismfilePtr= this;
   bottom_var->cache_data(kbp00);
   int bottom_var_id            = m_total_num_vars;
   m_variables[bottom_var_id]   = bottom_var;
   m_total_num_vars++;   
   // add node  x, y,norminal depth variables

   NativeSchismOutputVar * xVar        = new NativeSchismOutputVar(m_node_x);
   xVar->add_dimension(nNodeDim);  
   xVar->m_schismfilePtr   = this;
   xVar->cache_data(xcoord);
   delete xcoord; 
   int xVarId            = m_total_num_vars;
   m_variables[xVarId] = xVar;
   m_total_num_vars++;   


   NativeSchismOutputVar * yVar        = new NativeSchismOutputVar(m_node_y);
   yVar->add_dimension(nNodeDim);  
   yVar->m_schismfilePtr   = this;   
   yVar->cache_data(ycoord);
   delete ycoord; 
   int yVarId            = m_total_num_vars;
   m_variables[yVarId]   = yVar;
   m_total_num_vars++;   


   NativeSchismOutputVar * depthVar        = new NativeSchismOutputVar(m_node_depth);
   depthVar->add_dimension(nNodeDim);  
   depthVar->m_schismfilePtr   = this; 
   depthVar->cache_data(depth);
   delete depth; 
   delete floatArrayBuffer.cBuffer;
   delete singleIntBuffer.fourChar;

   int depthVarId            = m_total_num_vars;
   m_variables[depthVarId]   = depthVar;
   m_total_num_vars++;   

   union
   {
      int* intArray;
      char * cBuffer;
   } threeIntArrayBuffer;

   union
   {
      int* intArray;
      char * cBuffer;
   } fourIntArrayBuffer;

   singleIntBuffer.fourChar    = new char [DATA_NUM_BYTE];
   threeIntArrayBuffer.cBuffer = new char [3*DATA_NUM_BYTE];
   fourIntArrayBuffer.cBuffer  = new char [4*DATA_NUM_BYTE];

   int * nodes                    = new int [totalNumberOfEle*(MAX_CELL_NODE+1)];
   int index = 0;
   for(int iCell =0;iCell<totalNumberOfEle;iCell++)
   {
	  m_schismOutputFileStream->read(singleIntBuffer.fourChar,DATA_NUM_BYTE);
	  index=iCell*(MAX_CELL_NODE+1);
      int num_node_in_cell =  *singleIntBuffer.intValue;
	  if ((num_node_in_cell<3) || (num_node_in_cell>4))
	  {
		  std::string invalidCellNum("invalid cell number %i, must be 3 or 4",num_node_in_cell);
		  throw SCHISMFileException(invalidCellNum);
	  }
	  else if (num_node_in_cell==3)
	  {
		m_schismOutputFileStream->read(threeIntArrayBuffer.cBuffer,3*DATA_NUM_BYTE);
		nodes[index]   = num_node_in_cell;
        nodes[index+1] = threeIntArrayBuffer.intArray[0]; 
        nodes[index+2] = threeIntArrayBuffer.intArray[1]; 
		nodes[index+3] = threeIntArrayBuffer.intArray[2];
	  }
	  else if (num_node_in_cell==4)
	  {
		m_schismOutputFileStream->read(fourIntArrayBuffer.cBuffer,4*DATA_NUM_BYTE);
		nodes[index]   = num_node_in_cell;
        nodes[index+1] = fourIntArrayBuffer.intArray[0]; 
        nodes[index+2] = fourIntArrayBuffer.intArray[1]; 
		nodes[index+3] = fourIntArrayBuffer.intArray[2];
		nodes[index+4] = fourIntArrayBuffer.intArray[3];
	  }
   }   

   //momerize current location in file,used to navigated to
   //output data block

   std::streamoff  startOffset =  m_schismOutputFileStream->tellg();


   //
   int nCellDim                    = m_total_num_dims;
   SCHISMDim * cellDim              = newSCHISMDim(m_dim_mesh_faces,
                                                  nCellDim,
                                                  totalNumberOfEle);
   m_dimensions[nCellDim]          = cellDim;
   m_total_num_dims++;


   int nCellNodeDim                 = m_total_num_dims;
   SCHISMDim * cellNodeDim           = newSCHISMDim(MeshConstants::DIM_MESH_FACE_NODES,
                                                    nCellNodeDim,
                                                    (MAX_CELL_NODE+1));  // a extra position for cell number
   m_dimensions[nCellNodeDim]       = cellNodeDim;
   m_total_num_dims++;

   NativeSchismOutputVar * cellNodesVar        = new NativeSchismOutputVar(m_mesh_face_nodes);
   cellNodesVar->add_dimension(nCellDim);
   cellNodesVar->add_dimension(nCellNodeDim);
   cellNodesVar->m_schismfilePtr   = this;
   cellNodesVar->cache_data(nodes);

   delete nodes; 
   delete threeIntArrayBuffer.cBuffer;
   delete fourIntArrayBuffer.cBuffer;
   delete singleIntBuffer.fourChar;

   int cellNodeVarId          = m_total_num_vars;
   m_variables[cellNodeVarId] = cellNodesVar;
   m_total_num_vars++;    
   

   // built TIME dim and TIME var
   int nTIMEDim               = m_total_num_dims;
   SCHISMDim * TIMEDim         = newSCHISMDim(m_dim_time,
                                             nTIMEDim,
                                             nrec);
   m_dimensions[nTIMEDim]     = TIMEDim;
   m_total_num_dims++;

   NativeSchismOutputVar * TIMEVar         = new NativeSchismOutputVar(m_time);
   TIMEVar->add_dimension(nTIMEDim);
   TIMEVar->m_schismfilePtr   = this;
   int TIMEVarId              = m_total_num_vars;
   m_variables[TIMEVarId]     = TIMEVar;
   m_total_num_vars++;    

   // built surface and output data var
   NativeSchismOutputVar * surfaceVar       = new NativeSchismOutputVar(m_node_surface);
   surfaceVar->add_dimension(nTIMEDim);
   surfaceVar->add_dimension(nNodeDim);
   surfaceVar->m_schismfilePtr   = this;
   // surface data begin after TIME and step values in a data record block
   surfaceVar->setOffsetInDataBlock(2*DATA_NUM_BYTE);
   int surfaceVarId           = m_total_num_vars;
   m_variables[surfaceVarId]  = surfaceVar;
   m_total_num_vars++;    
   
   // compute a single step size based on ouput i23d
   int singleRecordByteSize = DATA_NUM_BYTE +                    // TIME
                              DATA_NUM_BYTE +                    // step
                              DATA_NUM_BYTE * totalNumberOfNodes; // eta2
  
  
   if (i23d==2)
   {
     singleRecordByteSize += ivs*totalNumberOfNodes*DATA_NUM_BYTE;
   }
   else
   {
     for(int iNode=0;iNode<totalNumberOfNodes;iNode++)
        {
          singleRecordByteSize += ivs*DATA_NUM_BYTE*(nvrt-max(1,kbp00[iNode])+1);
        }
   }                
 
  float* TIMEValues = new float [nrec];

  
  union
  {
     float * floatValue;
     char* fourChar;
  }  singleFloatBuffer;

  singleFloatBuffer.fourChar = new char [DATA_NUM_BYTE];   

  m_schismOutputFileStream->seekg(startOffset);

  for(int iRec =0; iRec<nrec;iRec++)
  {
     //m_schismOutputFileStream->seekg(startOffset+iRec*singleRecordByteSize);
     m_schismOutputFileStream->read(singleFloatBuffer.fourChar,DATA_NUM_BYTE);    
     TIMEValues[iRec] = *(singleFloatBuffer.floatValue);
     m_schismOutputFileStream->seekg(singleRecordByteSize-DATA_NUM_BYTE,ios::cur);
  }
 
  //move stream to begin 
  m_schismOutputFileStream->clear();
  m_schismOutputFileStream->seekg(0,ios::beg);
  
  
  // save dataBlockstart and datablock length as member data  
  m_dataBlockBeginOffset = startOffset;
  m_dataBlockLength      = singleRecordByteSize;
  
  TIMEVar->cache_data(TIMEValues);
  delete singleFloatBuffer.fourChar;
  delete TIMEValues;  

  
  
  //extract output variable name
  //startPos       = m_SCHISMOutputFile.find_last_of("_");
  //size_t endPos  = m_SCHISMOutputFile.find_last_of(".");
  char * fileDir = 0;
  char * fileName = new char [MAX_FILE_NAME_LEN];
  char * fileExt = 0;
  decomposePath(m_SCHISMOutputFile.c_str(), fileDir, fileName, fileExt);
  std::string fileNameStr(fileName);
  startPos       = fileNameStr.find_first_of("_");
  std::string SCHISMOutput = "output";   

 // if (!(startPos == std::string::npos)&&!(endPos == std::string::npos))
   if (!(startPos == std::string::npos))
  {
     //SCHISMOutput  = m_SCHISMOutputFile.substr(startPos+1,endPos-startPos-1);
	 SCHISMOutput  = fileNameStr.substr(startPos+1);
  }
  else
  {
     cout<<"file name "<<m_SCHISMOutputFile<<" don't contain valid data name\n";
  }

  std::string labelName   ;
  if (m_varLongNameMap.find(SCHISMOutput)!=m_varLongNameMap.end())
  {
     labelName = m_varLongNameMap[SCHISMOutput];
  }
 else
  {
     labelName = SCHISMOutput;
  }
  delete fileName;
  NativeSchismOutputVar * outputVar       = new NativeSchismOutputVar(labelName);
  outputVar->add_dimension(nTIMEDim);
  outputVar->add_dimension(nNodeDim);
  outputVar->m_schismfilePtr   = this;
  outputVar->setOffsetInDataBlock(DATA_NUM_BYTE * (totalNumberOfNodes+2));


  
  //outputVar->add_att(center_att);

  // decide if it has layer dimension
  if(i23d == 3)
  {
    outputVar->add_dimension(nvDim);
    // also setup staggered layering
    //outputVar->set_kbp(kbp00);
  }

 // is vector
 if(ivs>1) 
 {
    outputVar->add_dimension(nComponentDim);
 }
    
 int outputVarId            = m_total_num_vars;
 m_variables[outputVarId]   = outputVar;
 m_total_num_vars++;     
 delete kbp00; 
 return true; 
}


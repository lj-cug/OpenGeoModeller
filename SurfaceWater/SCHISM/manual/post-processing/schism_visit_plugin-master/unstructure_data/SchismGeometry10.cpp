#include "SchismGeometry10.h"
#include "MeshConstants10.h"


int max_node_in_cell = MeshConstants10::MAX_NUM_NODE_PER_CELL;

long invalid_id = MeshConstants10::INVALID_NUM;

int local_3side_node[3][2]={{1,2},
                          {2,0},
                          {0,1}};

int local_4side_node[4][2]={{1,2},
                          {2,3},
                          {3,0},
                          {0,1}};


long meshSideNum(long       *  a_meshNode,
                 const long &  a_numberEle,
				 const long &  a_numberNode)
{
	long * numEleAtNode = new long [a_numberNode];
	for(long i=0;i<a_numberNode;i++)
	{
		numEleAtNode[i]=0;
	}

    for(long iele=0;iele<a_numberEle;iele++)
	{
		int numNode = a_meshNode[iele*(max_node_in_cell+1)];
        for (long iNode=0;iNode<numNode;iNode++)
		{
          long node=a_meshNode[iele*(max_node_in_cell+1)+1+iNode]-1;
          numEleAtNode[node]+=1;
		}
	}
    
	int maxEleAtNode = 0;

	for(long i=0;i<a_numberNode;i++)
	{
		if (numEleAtNode[i]>maxEleAtNode)
		{
			maxEleAtNode = numEleAtNode[i];
		}
	}

	long * eleAtNode = new long  [maxEleAtNode*a_numberNode];

	for(long i=0;i<maxEleAtNode*a_numberNode;i++)
	{
		eleAtNode[i]=invalid_id;
	}

	for(long i=0;i<a_numberNode;i++)
	{
		numEleAtNode[i]=0;
	}

	for(long iele=0;iele<a_numberEle;iele++)
	{
		int numNode = a_meshNode[iele*(max_node_in_cell+1)];
        for (long iNode=0;iNode<numNode;iNode++)
		{
          long node=a_meshNode[iele*(max_node_in_cell+1)+1+iNode]-1;
		  long  index = node*maxEleAtNode+numEleAtNode[node];
		  eleAtNode[index]=iele;
          numEleAtNode[node]+=1;
		}
	}

	long * neiborEleAllSide = new long [max_node_in_cell*a_numberEle];
	for(long i=0;i<max_node_in_cell*a_numberEle;i++)
	{
		neiborEleAllSide[i]=invalid_id;
	}
	for(long iele=0;iele<a_numberEle;iele++)
	{
		int numNode = a_meshNode[iele*(max_node_in_cell+1)];
		int numSide = numNode;
        for (int iSide=0;iSide<numSide;iSide++)
		{
		  long node1=invalid_id;
		  long node2=invalid_id;
		  long idStart = iele*(max_node_in_cell+1)+1;
          if(numSide==3)
		  {
			  node1= a_meshNode[idStart+local_3side_node[iSide][0]]-1;
			  node2= a_meshNode[idStart+local_3side_node[iSide][1]]-1;
		  }
		  else
		  {
			  node1= a_meshNode[idStart+local_4side_node[iSide][0]]-1;
			  node2= a_meshNode[idStart+local_4side_node[iSide][1]]-1;
		  }

		  for(int iNeiborEle=0;iNeiborEle<numEleAtNode[node1];iNeiborEle++)
		  {
			  long eleIndex = eleAtNode[node1*maxEleAtNode+iNeiborEle];
			  if ((eleIndex>invalid_id)&&(eleIndex != iele))
			  {
				  long eleStartIndex = eleIndex*(max_node_in_cell+1);
				  int numNode = a_meshNode[eleStartIndex];
				  bool shareThisSide = false;
				  for(int iiNode=0;iiNode<numNode;iiNode++)
				  {
					  long nodeIndex = a_meshNode[eleStartIndex+1+iiNode]-1;
					  if (nodeIndex==node2)
					  {
						  shareThisSide = true;
						  break;
					  }

				  }
				  if (shareThisSide)
				  {
					  neiborEleAllSide[max_node_in_cell*iele+iSide]=eleIndex;
				  }
			  }
		  }
		}
	}

	long totalNumSide =0;

	for(long iele=0;iele<a_numberEle;iele++)
	{
		int numNode = a_meshNode[iele*(max_node_in_cell+1)];
		int numSide = numNode;
        for (int iSide=0;iSide<numSide;iSide++)
		{
			long neiborEle = neiborEleAllSide[max_node_in_cell*iele+iSide];
			if ((neiborEle == invalid_id) || (iele<neiborEle))
			{
				totalNumSide++;
			}

		}

	}
	delete numEleAtNode;
	delete eleAtNode;
	delete neiborEleAllSide;
	return totalNumSide;
}

void  meshSideNodeNeiborEle(long    *      a_mesh_node,
                            long    *     &a_node_neighbor_ele,
							long    *     &a_side_neighbor_ele,
							int     &      a_max_ele_at_node,
							const long &  a_number_ele,
				            const long &  a_number_node,
							const long &  a_number_side)
{

	int * num_ele_at_node = new int [a_number_node];
	for(long i=0;i<a_number_node;i++)
	{
		num_ele_at_node[i]=0;
	}

    for(long iele=0;iele<a_number_ele;iele++)
	{
		int numNode = a_mesh_node[iele*(max_node_in_cell+1)];
        for (long iNode=0;iNode<numNode;iNode++)
		{
          long node=a_mesh_node[iele*(max_node_in_cell+1)+1+iNode]-1;
          num_ele_at_node[node]+=1;
		}
	}
    
	int maxEleAtNode = 0;

	for(long i=0;i<a_number_node;i++)
	{
		if (num_ele_at_node[i]>maxEleAtNode)
		{
			maxEleAtNode = num_ele_at_node[i];
		}
	}
	a_max_ele_at_node=maxEleAtNode;

	a_node_neighbor_ele = new long  [maxEleAtNode*a_number_node];

	for(long i=0;i<maxEleAtNode*a_number_node;i++)
	{
		a_node_neighbor_ele[i]=invalid_id;
	}

	//set num_ele_at_node to zero for the usage
	//of next step to place node neighbor element 
	//although we already computed this data
	for(long i=0;i<a_number_node;i++)
	{
		num_ele_at_node[i]=0;
	}


	for(long iele=0;iele<a_number_ele;iele++)
	{
		int numNode = a_mesh_node[iele*(max_node_in_cell+1)];
        for (long iNode=0;iNode<numNode;iNode++)
		{
          long node=a_mesh_node[iele*(max_node_in_cell+1)+1+iNode]-1;
		  long  index = node*maxEleAtNode+num_ele_at_node[node];
		  a_node_neighbor_ele[index]=iele;
          num_ele_at_node[node]+=1;
		}
	}

	a_side_neighbor_ele = new long [2*a_number_side]; //only max 2 element neibor a side
	long * ele_side_neighbor_ele=new long [a_number_ele*max_node_in_cell];

	for(long i=0;i<2*a_number_side;i++)
	{
		a_side_neighbor_ele[i]=invalid_id;
	}

	for(long i=0;i<a_number_ele*max_node_in_cell;i++)
	{
		ele_side_neighbor_ele[i]=invalid_id;
	}
	for(long iele=0;iele<a_number_ele;iele++)
	{
		int numNode = a_mesh_node[iele*(max_node_in_cell+1)];
		int numSide = numNode;
        for (int iSide=0;iSide<numSide;iSide++)
		{
		  long node1=invalid_id;
		  long node2=invalid_id;
		  long idStart = iele*(max_node_in_cell+1)+1;
          if(numSide==3)
		  {
			  node1= a_mesh_node[idStart+local_3side_node[iSide][0]]-1;
			  node2= a_mesh_node[idStart+local_3side_node[iSide][1]]-1;
		  }
		  else
		  {
			  node1= a_mesh_node[idStart+local_4side_node[iSide][0]]-1;
			  node2= a_mesh_node[idStart+local_4side_node[iSide][1]]-1;
		  }

		  for(int iNeiborEle=0;iNeiborEle<num_ele_at_node[node1];iNeiborEle++)
		  {
			  long eleIndex = a_node_neighbor_ele[node1*maxEleAtNode+iNeiborEle];
			  if ((eleIndex>invalid_id)&&(eleIndex != iele))
			  {
				  long eleStartIndex = eleIndex*(max_node_in_cell+1);
				  int numNode = a_mesh_node[eleStartIndex];
				  bool shareThisSide = false;
				  for(int iiNode=0;iiNode<numNode;iiNode++)
				  {
					  long nodeIndex = a_mesh_node[eleStartIndex+1+iiNode]-1;
					  if (nodeIndex==node2)
					  {
						  shareThisSide = true;
						  break;
					  }

				  }
				  if (shareThisSide)
				  {
					  ele_side_neighbor_ele[max_node_in_cell*iele+iSide]=eleIndex;
				  }
			  }
		  }
		}
	}


	long ns=0;

	for(long iele=0;iele<a_number_ele;iele++)
	{
		int numNode = a_mesh_node[iele*(max_node_in_cell+1)];
		int numSide = numNode;
        for (int iSide=0;iSide<numSide;iSide++)
		{
		  long node1=invalid_id;
		  long node2=invalid_id;
		  long idStart = iele*(max_node_in_cell+1)+1;
          if(numSide==3)
		  {
			  node1= a_mesh_node[idStart+local_3side_node[iSide][0]]-1;
			  node2= a_mesh_node[idStart+local_3side_node[iSide][1]]-1;
		  }
		  else
		  {
			  node1= a_mesh_node[idStart+local_4side_node[iSide][0]]-1;
			  node2= a_mesh_node[idStart+local_4side_node[iSide][1]]-1;
		  }

		  long neibor_ele=ele_side_neighbor_ele[max_node_in_cell*iele+iSide];

		  if (( neibor_ele==invalid_id)||(iele< neibor_ele))
		  {
			  a_side_neighbor_ele[ns*2]=iele;
			  a_side_neighbor_ele[ns*2+1]= neibor_ele;
			  ns++;
		  }
		}
	}


	delete num_ele_at_node;
	delete ele_side_neighbor_ele;

}

void   meshSideNode(long    *     a_sideNode,
				    long    *     a_meshNode,
					const long &  a_numberSide,
                    const long &  a_numberEle,
					const long &  a_numberNode)

{
	int * numEleAtNode = new int [a_numberNode];
	for(long i=0;i<a_numberNode;i++)
	{
		numEleAtNode[i]=0;
	}

    for(long iele=0;iele<a_numberEle;iele++)
	{
		int numNode = a_meshNode[iele*(max_node_in_cell+1)];
        for (int iNode=0;iNode<numNode;iNode++)
		{
          long node=a_meshNode[iele*(max_node_in_cell+1)+1+iNode]-1;
          numEleAtNode[node]+=1;
		}
	}
    
	int maxEleAtNode = 0;

	for(long i=0;i<a_numberNode;i++)
	{
		if (numEleAtNode[i]>maxEleAtNode)
		{
			maxEleAtNode = numEleAtNode[i];
		}
	}

	long * eleAtNode = new long  [maxEleAtNode*a_numberNode];

	for(long i=0;i<maxEleAtNode*a_numberNode;i++)
	{
		eleAtNode[i]=invalid_id;
	}

	for(long i=0;i<a_numberNode;i++)
	{
		numEleAtNode[i]=0;
	}

	for(long iele=0;iele<a_numberEle;iele++)
	{
		int numNode = a_meshNode[iele*(max_node_in_cell+1)];
        for (int iNode=0;iNode<numNode;iNode++)
		{
          long node=a_meshNode[iele*(max_node_in_cell+1)+1+iNode]-1;
		  long  index = node*maxEleAtNode+numEleAtNode[node];
		  eleAtNode[index]=iele;
          numEleAtNode[node]+=1;
		}
	}

	long * neiborEleAllSide = new long [max_node_in_cell*a_numberEle];
	for(long i=0;i<max_node_in_cell*a_numberEle;i++)
	{
		neiborEleAllSide[i]=invalid_id;
	}
	for(long iele=0;iele<a_numberEle;iele++)
	{
		int numNode = a_meshNode[iele*(max_node_in_cell+1)];
		int numSide = numNode;
        for (long iSide=0;iSide<numSide;iSide++)
		{
		  long node1=invalid_id;
		  long node2=invalid_id;
		  long idStart = iele*(max_node_in_cell+1)+1;
          if(numSide==3)
		  {
			  node1= a_meshNode[idStart+local_3side_node[iSide][0]]-1;
			  node2= a_meshNode[idStart+local_3side_node[iSide][1]]-1;
		  }
		  else
		  {
			  node1= a_meshNode[idStart+local_4side_node[iSide][0]]-1;
			  node2= a_meshNode[idStart+local_4side_node[iSide][1]]-1;
		  }

		  for(int iNeiborEle=0;iNeiborEle<numEleAtNode[node1];iNeiborEle++)
		  {
			  long eleIndex = eleAtNode[node1*maxEleAtNode+iNeiborEle];
			  if((eleIndex>invalid_id)&&(eleIndex != iele))
			  {
				  long eleStartIndex = eleIndex*(max_node_in_cell+1);
				  int numNode = a_meshNode[eleStartIndex];
				  bool shareThisSide = false;
				  for(int iiNode=0;iiNode<numNode;iiNode++)
				  {
					  long nodeIndex = a_meshNode[eleStartIndex+1+iiNode]-1;
					  if (nodeIndex==node2)
					  {
						  shareThisSide = true;
						  break;
					  }

				  }
				  if (shareThisSide)
				  {
					  neiborEleAllSide[max_node_in_cell*iele+iSide]=eleIndex;
				  }
			  }
		  }
		}
	}

	
	

	long totalNumSideTemp =0;
	
	for(long iele=0;iele<a_numberEle;iele++)
	{
		int numNode = a_meshNode[iele*(max_node_in_cell+1)];
		int numSide = numNode;
        for (int iSide=0;iSide<numSide;iSide++)
		{
			long node1=invalid_id;
		    long node2=invalid_id;
		    long idStart = iele*(max_node_in_cell+1)+1;
            if(numSide==3)
		    {
			    node1= a_meshNode[idStart+local_3side_node[iSide][0]]-1;
			    node2= a_meshNode[idStart+local_3side_node[iSide][1]]-1;
		    }
		    else
		    {
			    node1= a_meshNode[idStart+local_4side_node[iSide][0]]-1;
			    node2= a_meshNode[idStart+local_4side_node[iSide][1]]-1;
		    }
			long neiborEle = neiborEleAllSide[max_node_in_cell*iele+iSide];
			if ((neiborEle == invalid_id) || (iele<neiborEle)) // a new side
			{
				a_sideNode[2*totalNumSideTemp]  =node1;
				a_sideNode[2*totalNumSideTemp+1]=node2;
				totalNumSideTemp++;
			}
		}
	}

	

	delete numEleAtNode;
	delete eleAtNode;
	delete neiborEleAllSide;



}
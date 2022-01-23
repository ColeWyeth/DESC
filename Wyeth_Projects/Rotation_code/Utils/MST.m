function R_est = MST(Ind, RijMat, SVec)

    Ind_i = Ind(:,1);
    Ind_j = Ind(:,2);
   
    n=max(Ind,[],'all');
    m=size(Ind_i,1); % number of edges
    
    for l = 1:m
        i=Ind_i(l);j=Ind_j(l);
        IndMat(i,j)=l;
        IndMat(j,i)=-l;
    end
    
    disp('build spanning tree');

    Indfull_i = [Ind_i;Ind_j];
    Indfull_j = [Ind_j;Ind_i];
    Sfull = [SVec, SVec];
    %Sfull = [exp(SVec.^2), exp(SVec.^2)];
    DG = sparse(Indfull_i,Indfull_j,Sfull);
    [tree,~]=graphminspantree(DG);
    [T1, T2, ~] = find(tree);
    sizetree=size(T1,1);
    AdjTree = zeros(n);
    for k=1:sizetree
        i=T1(k); j=T2(k);
        AdjTree(i,j)=1;
        AdjTree(j,i)=1;
    end
    %[~, rootnodes]=max(sum(AdjTree));
    rootnodes = 1;
    added=zeros(1,n);
    R_est = zeros(3,3,n);
    R_est(:,:,rootnodes)=eye(3);
    added(rootnodes)=1;
    newroots = [];
    while sum(added)<n
        for node_root = rootnodes
            leaves = find((AdjTree(node_root,:).*(1-added))==1);
            newroots = [newroots, leaves];
            for node_leaf=leaves
                edge_leaf = IndMat(node_leaf,node_root);
                if edge_leaf>0
                    R_est(:,:,node_leaf)=RijMat(:,:,abs(edge_leaf))*R_est(:,:,node_root);
                else
                    R_est(:,:,node_leaf)=(RijMat(:,:,abs(edge_leaf)))'*R_est(:,:,node_root);
                end
                added(node_leaf)=1;
            end
        end
        rootnodes = newroots;
    end

end
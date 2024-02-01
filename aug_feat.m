sep = 10000;

% txt 파일로부터 데이터 불러오기
scal = '20';
data = dlmread('graphs/scal_20.txt');  % dlmread 함수는 공백 또는 탭으로 구분된 파일을 읽을 수 있습니다.


% 데이터에서 행, 열, 값 추출
nodes = data(1,1);
edges = data(1,2);
rows = data(2:end, 1);
cols = data(2:end, 2);
values = data(2:end, 3);

format long

% sparse matrix 생성
sparse_matrix = sparse(rows+1, cols+1, values, nodes, nodes);

fin = transpose(block_partition(sparse_matrix,sep));

writematrix(fin, sprintf('node_feature_%s_LT.txt', scal),'Delimiter','tab')
type filename


function result = block_partition(matrix, block_size)
    [rows, ~] = size(matrix);
    row_blocks = floor(rows / block_size);
    total_elapsed_time_cycle = 0;
    rr = zeros(10, rows);
    
    for i = 1:(row_blocks + 1)
        local_matrix = matrix;
        if i * block_size > rows
            block = local_matrix((i - 1) * block_size + 1:end, :);
        else
            block = local_matrix((i - 1) * block_size + 1:i * block_size, :);
        end
        
        start1 = tic;
        for j = 1:10
            block = block * local_matrix;
            if i*block_size > rows
                rr(j,(i-1)*block_size+1:end) = diag(block, (i-1) * block_size)';
            else
                rr(j,(i-1)*block_size+1:i*block_size) = diag(block, (i-1) * block_size)';
            end
        end
        end1 = toc(start1);
        total_elapsed_time_cycle = total_elapsed_time_cycle + end1;
    end
    disp(total_elapsed_time_cycle);
    result = rr;
end

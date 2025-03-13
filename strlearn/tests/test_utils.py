from strlearn.streams import DataStream


class StreamSubset(DataStream):
    def __init__(self, base_stream: DataStream, yield_n_chunks=5):
        super().__init__()
        self.base_stream = base_stream
        self.yield_n_chunks = yield_n_chunks
        self.chunks_used = 0

    def get_chunk(self):
        self.chunks_used += 1
        return self.base_stream.get_chunk()

    def is_dry(self):
        return self.chunks_used == self.yield_n_chunks or self.base_stream.is_dry()

    @property
    def n_chunks(self):
        return self.base_stream.n_chunks

    @property
    def previous_chunk(self):
        return self.base_stream.previous_chunk

    @property
    def classes_(self):
        return self.base_stream.classes_

    @property
    def chunk_id(self):
        return self.base_stream.chunk_id
